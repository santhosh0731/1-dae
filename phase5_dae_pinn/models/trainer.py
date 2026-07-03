"""
DAE-PINN Training Engine (Corrected for Scaling & Physical Units)
==================================================================
Core training loop integrating 3-phase curriculum, NTK adaptive loss weights,
and the differentiable Radau-IIA integration residual layer, with targets
properly denormalized to physical units.
"""

import sys
import time
import logging
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from phase5_dae_pinn.models.dae_pinn import DAEPINNModel
from phase5_dae_pinn.irk.embedded_irk import DifferentiableRadauIIALayer
from phase5_dae_pinn.physics.residuals import compute_all_residuals
from phase5_dae_pinn.adaptive_loss.scheduler import PhaseCurriculumScheduler
from phase5_dae_pinn.adaptive_loss.weighting import NTKGradientBalancing

logger = logging.getLogger(__name__)


class DAEPINNTrainer:
    """Trainer manager for DAE-PINN models."""

    def __init__(
        self,
        model: DAEPINNModel,
        config: dict,
        ckpt_dir: Path,
        log_dir: Path,
        scalers: dict,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = Path(ckpt_dir)
        self.log_dir = Path(log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config['training']
        self.epochs = train_cfg['epochs']
        self.patience = train_cfg['patience']
        self.batch_size = train_cfg['batch_size']

        self.optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=train_cfg['lr_min']
        )

        # Embedded IRK layer
        self.irk_layer = DifferentiableRadauIIALayer(
            model=model,
            step_size_h=config['irk'].get('step_size_h', 1.0e-5)
        ).to(self.device)

        # Target denormalization constants
        self.mean_Y = torch.tensor(scalers['Y'].mean_[:3], dtype=torch.float32, device=self.device)
        self.std_Y = torch.tensor(scalers['Y'].scale_[:3], dtype=torch.float32, device=self.device)
        self.t_std = float(scalers['X'].scale_[0])
        self.mean_t = float(scalers['X'].mean_[0])
        self.physics_scales = scalers.get('physics_scales', None)

        # Curriculum & NTK
        self.curriculum = PhaseCurriculumScheduler(config)
        self.ntk_balancer = NTKGradientBalancing(
            model=model,
            alpha=config['curriculum']['phase_c'].get('ema_alpha', 0.9),
            update_freq=10,
        )

        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.history: Dict[str, list] = {
            'epoch': [], 'train_total': [], 'val_total': [],
            'data': [], 'kvl': [], 'kcl': [], 'dae': [], 'irk': [],
            'bc': [], 'ic': [], 'pwr': [], 'nrg': [],
            'lambda_kvl': [], 'lambda_kcl': [], 'lambda_irk': [], 'lr': []
        }

    def compute_loss(
        self,
        pred: torch.Tensor,
        target_phys: torch.Tensor, # Raw physical target states [Vout, IL, Vc]
        params: torch.Tensor,      # Raw physical inputs [Vin, D, L, C, Rload]
        inputs: torch.Tensor,      # Normalized input features
        weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the weighted 9-component loss in physical space."""
        loss_dict = {}

        # 1. Data loss - Stage 9: Scale errors by standard deviation (equivalent to normalized MSE)
        loss_dict['data'] = torch.mean(((pred[:, :3] - target_phys) / self.std_Y) ** 2)

        # 2. Physics residuals (KVL, KCL, DAE, PWR, NRG) - Stage 8: Normalize using dataset reference scales
        phys = compute_all_residuals(pred, params, self.physics_scales)
        loss_dict.update(phys)

        # 3. Embedded IRK loss (with normalized inputs and scaled time std)
        irk_res = self.irk_layer(inputs, pred, params, self.t_std)
        loss_dict['irk'] = torch.mean(irk_res ** 2)

        # 4. Boundary and initial conditions - Stage 10: Identify t=0 and t=T points in the shuffled batch
        t_raw = inputs[:, 0] * self.t_std + self.mean_t
        ic_mask = (t_raw < 1e-7)
        bc_mask = (t_raw > 0.00499)

        if ic_mask.any():
            loss_dict['ic'] = torch.mean(((pred[ic_mask, :3] - target_phys[ic_mask]) / self.std_Y) ** 2)
        else:
            loss_dict['ic'] = torch.tensor(0.0, device=self.device)

        if bc_mask.any():
            loss_dict['bc'] = torch.mean(((pred[bc_mask, :3] - target_phys[bc_mask]) / self.std_Y) ** 2)
        else:
            loss_dict['bc'] = torch.tensor(0.0, device=self.device)

        # Weighted total
        total = torch.tensor(0.0, device=self.device)
        for k, val in loss_dict.items():
            w = weights.get(k, 0.0)
            total = total + w * val

        loss_dict['total'] = total
        return total, loss_dict

    def train_epoch(self, train_loader, weights: Dict[str, float]) -> Dict:
        self.model.train()
        epoch_losses = {k: 0.0 for k in ['total','data','kvl','kcl','dae','irk','bc','ic','pwr','nrg']}
        n_batches = 0

        for X_batch, Y_batch, params_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            params_batch = params_batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X_batch)

            # Denormalize target to physical space
            Y_batch_phys = Y_batch[:, :3] * self.std_Y + self.mean_Y

            total, loss_dict = self.compute_loss(pred, Y_batch_phys, params_batch, X_batch, weights)
            total.backward()

            # Stage 12: Monitor gradients and check for NaNs
            grad_norms = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.detach().data.norm(2).item())
            if grad_norms:
                max_grad = max(grad_norms)
                min_grad = min(grad_norms)
                mean_grad = sum(grad_norms) / len(grad_norms)
            else:
                max_grad, min_grad, mean_grad = 0.0, 0.0, 0.0

            if np.isnan(total.item()) or (grad_norms and np.isnan(mean_grad)):
                logger.error("  [FATAL] NaN detected in loss or gradients! Aborting training.")
                sys.exit(1)

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def val_epoch(self, val_loader, weights: Dict[str, float]) -> Dict:
        self.model.eval()
        epoch_losses = {k: 0.0 for k in ['total','data','kvl','kcl','dae','irk','bc','ic','pwr','nrg']}
        n_batches = 0

        for X_batch, Y_batch, params_batch in val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            params_batch = params_batch.to(self.device)

            pred = self.model(X_batch)

            # Denormalize target to physical space
            Y_batch_phys = Y_batch[:, :3] * self.std_Y + self.mean_Y

            total, loss_dict = self.compute_loss(pred, Y_batch_phys, params_batch, X_batch, weights)

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = self.ckpt_dir / f"dae_pinn_{tag}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }, path)
        logger.info(f"  [CKPT] Saved: {path.name} (epoch={epoch}, val={val_loss:.6f})")

    def _log_epoch(self, epoch, tr, vl, weights, lr):
        self.history['epoch'].append(epoch)
        self.history['train_total'].append(tr['total'])
        self.history['val_total'].append(vl['total'])
        for k in ['data','kvl','kcl','dae','irk','bc','ic','pwr','nrg']:
            self.history[k].append(tr.get(k, 0.0))
        self.history['lambda_kvl'].append(weights.get('kvl', 0.0))
        self.history['lambda_kcl'].append(weights.get('kcl', 0.0))
        self.history['lambda_irk'].append(weights.get('irk', 0.0))
        self.history['lr'].append(lr)

    def fit(self, train_loader, val_loader) -> Dict:
        log_csv = self.log_dir / "training_log.csv"
        fieldnames = ['epoch','phase','train_total','val_total','data','kvl','kcl','dae','irk',
                      'bc','ic','pwr','nrg','lambda_kvl','lambda_kcl','lambda_irk','lr','time_s']

        logger.info("=" * 70)
        logger.info("  PHASE 5 — DAE-PINN TRAINING RUN")
        logger.info(f"  Device: {self.device} | Epochs: {self.epochs} | Batch: {self.batch_size}")
        logger.info(f"  Parameters: {self.model.count_parameters():,}")
        logger.info("=" * 70)

        with open(log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                phase_name = self.curriculum.phase_name(epoch)

                # Reset early stopping trackers on phase transitions to prevent prematurely stopping when physics constraints are activated
                if epoch == self.config['curriculum']['phase_a']['end_epoch'] + 1 or \
                   epoch == self.config['curriculum']['phase_b']['end_epoch'] + 1:
                    logger.info(f"  [Phase Transition] Resetting early stopping tracker for {phase_name}...")
                    self.best_val_loss = float('inf')
                    self.no_improve = 0

                # Fetch loss weights
                weights = self.curriculum.get_fixed_weights(epoch)

                # In Phase C, apply NTK balancing
                if epoch > self.config['curriculum']['phase_b']['end_epoch']:
                    # Build dummy tensors to query balancing on parameters
                    self.model.train()
                    X_dum, Y_dum, P_dum = next(iter(train_loader))
                    X_dum = X_dum[:64].to(self.device)
                    Y_dum = Y_dum[:64].to(self.device)
                    P_dum = P_dum[:64].to(self.device)
                    self.optimizer.zero_grad()
                    pred_dum = self.model(X_dum)

                    Y_dum_phys = Y_dum[:64, :3] * self.std_Y + self.mean_Y

                    _, l_dict = self.compute_loss(pred_dum, Y_dum_phys, P_dum, X_dum, weights)
                    phys_losses = {k: v for k, v in l_dict.items() if k not in ('total', 'data')}
                    balanced_weights = self.ntk_balancer.update(phys_losses, l_dict['data'])
                    weights.update(balanced_weights)

                # Train and validate
                tr = self.train_epoch(train_loader, weights)
                vl = self.val_epoch(val_loader, weights)

                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']

                self._log_epoch(epoch, tr, vl, weights, lr)
                elapsed = time.time() - t0

                writer.writerow({
                    'epoch': epoch, 'phase': phase_name,
                    'train_total': f"{tr['total']:.6f}",
                    'val_total': f"{vl['total']:.6f}",
                    'data': f"{tr['data']:.6f}",
                    'kvl': f"{tr['kvl']:.6f}",
                    'kcl': f"{tr['kcl']:.6f}",
                    'dae': f"{tr['dae']:.6f}",
                    'irk': f"{tr['irk']:.6f}",
                    'bc': f"{tr['bc']:.6f}",
                    'ic': f"{tr['ic']:.6f}",
                    'pwr': f"{tr['pwr']:.6f}",
                    'nrg': f"{tr['nrg']:.6f}",
                    'lambda_kvl': f"{weights.get('kvl',0.0):.4f}",
                    'lambda_kcl': f"{weights.get('kcl',0.0):.4f}",
                    'lambda_irk': f"{weights.get('irk',0.0):.4f}",
                    'lr': f"{lr:.2e}",
                    'time_s': f"{elapsed:.2f}",
                })
                f.flush()

                # Logger print every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"  Ep {epoch:4d} | {phase_name:25s} | "
                        f"Train: {tr['total']:.5f} | Val: {vl['total']:.5f} | "
                        f"IRK: {tr['irk']:.5f} | DAE: {tr['dae']:.5f} | "
                        f"LR: {lr:.2e}"
                    )

                # Validation early stopping on data prediction loss
                val_track = vl['data']
                if val_track < self.best_val_loss:
                    self.best_val_loss = val_track
                    self.no_improve = 0
                    self.save_checkpoint(epoch, vl['total'], tag="best")
                else:
                    self.no_improve += 1

                if self.no_improve >= self.patience:
                    logger.info(f"  [Early Stop] No improvement for {self.patience} epochs.")
                    self.save_checkpoint(epoch, vl['total'], tag="last")
                    break

        self.save_checkpoint(self.epochs, self.best_val_loss, tag="final")

        # Stage 15: Restore best validation checkpoint weights before returning
        best_path = self.ckpt_dir / "dae_pinn_best.pt"
        if best_path.exists():
            logger.info(f"  [CKPT] Restoring best validation model weights from {best_path.name}...")
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])

        return self.history
