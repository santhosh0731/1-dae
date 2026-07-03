"""
PINN Trainer
=============
Full training loop with:
  - 3-phase curriculum (Phase A → B → C)
  - Adaptive gradient-balanced loss weights (Phase C)
  - Adam + cosine LR annealing
  - Early stopping
  - Per-epoch CSV logging
  - Best checkpoint saving
"""

import sys
import time
import logging
import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from phase4_pinn.models.pinn_model import PINNModel
from phase4_pinn.models.loss_functions import PINNLoss
from phase4_pinn.models.adaptive_loss.weight_scheduler import (
    CurriculumScheduler, GradientBalancedWeightScheduler
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINNTrainer:
    """Complete PINN training manager."""

    def __init__(
        self,
        model:       PINNModel,
        config:      dict,
        ckpt_dir:    Path,
        log_dir:     Path,
    ):
        self.model    = model.to(DEVICE)
        self.config   = config
        self.ckpt_dir = Path(ckpt_dir)
        self.log_dir  = Path(log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config['training']
        self.epochs     = train_cfg['epochs']
        self.patience   = train_cfg['patience']
        self.batch_size = train_cfg['batch_size']

        self.criterion = PINNLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=train_cfg['lr_min']
        )

        # Curriculum + adaptive weight schedulers
        self.curriculum = CurriculumScheduler(config)
        self.grad_scheduler = GradientBalancedWeightScheduler(
            model=model,
            alpha=config['curriculum']['phase_c'].get('ema_alpha', 0.9),
            update_freq=10,
        )
        self.curriculum.set_adaptive_scheduler(self.grad_scheduler)

        # Tracking
        self.best_val_loss = float('inf')
        self.no_improve    = 0
        self.history: Dict[str, list] = {
            'epoch': [], 'train_total': [], 'val_total': [],
            'data': [], 'kvl': [], 'kcl': [], 'dae': [],
            'bc': [], 'ic': [], 'pwr': [],
            'lambda_kvl': [], 'lambda_kcl': [], 'lr': []
        }

    def train_epoch(self, train_loader, weights: Dict[str, float]) -> Dict:
        self.model.train()
        epoch_losses = {k: 0.0 for k in ['total','data','kvl','kcl','dae','bc','ic','pwr']}
        n_batches = 0

        for X_batch, Y_batch, params_batch in train_loader:
            X_batch      = X_batch.to(DEVICE)
            Y_batch      = Y_batch.to(DEVICE)
            params_batch = params_batch.to(DEVICE)

            self.optimizer.zero_grad()
            pred = self.model(X_batch)

            total, loss_dict = self.criterion.total_loss(
                pred, Y_batch[:, :3], params_batch,
                weights=weights
            )
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def val_epoch(self, val_loader, weights: Dict[str, float]) -> Dict:
        self.model.eval()
        epoch_losses = {k: 0.0 for k in ['total','data','kvl','kcl','dae','bc','ic','pwr']}
        n_batches = 0

        for X_batch, Y_batch, params_batch in val_loader:
            X_batch      = X_batch.to(DEVICE)
            Y_batch      = Y_batch.to(DEVICE)
            params_batch = params_batch.to(DEVICE)

            pred = self.model(X_batch)
            total, loss_dict = self.criterion.total_loss(
                pred, Y_batch[:, :3], params_batch,
                weights=weights
            )
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = self.ckpt_dir / f"pinn_{tag}.pt"
        torch.save({
            'epoch':      epoch,
            'model_state': self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'val_loss':   val_loss,
            'config':     self.config,
        }, path)
        logger.info(f"  [CKPT] Saved: {path.name} (epoch={epoch}, val={val_loss:.6f})")

    def _log_epoch(self, epoch, tr, vl, weights, lr):
        """Store history and write CSV row."""
        self.history['epoch'].append(epoch)
        self.history['train_total'].append(tr['total'])
        self.history['val_total'].append(vl['total'])
        for k in ['data','kvl','kcl','dae','bc','ic','pwr']:
            self.history[k].append(tr.get(k, 0.0))
        self.history['lambda_kvl'].append(weights.get('kvl', 0.0))
        self.history['lambda_kcl'].append(weights.get('kcl', 0.0))
        self.history['lr'].append(lr)

    def fit(self, train_loader, val_loader) -> Dict:
        """Main training loop."""
        log_csv = self.log_dir / "training_log.csv"
        fieldnames = ['epoch','phase','train_total','val_total','data',
                      'kvl','kcl','dae','bc','ic','pwr',
                      'lambda_kvl','lambda_kcl','lr','time_s']

        logger.info("=" * 65)
        logger.info("  PHASE 4 — PINN TRAINING")
        logger.info(f"  Device: {DEVICE} | Epochs: {self.epochs} | Batch: {self.batch_size}")
        logger.info(f"  Parameters: {self.model.count_parameters():,}")
        logger.info("=" * 65)

        with open(log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                phase_name = self.curriculum.phase_name(epoch)

                # Get current loss weights
                weights = self.curriculum.get_weights(epoch)

                # Train + validate
                tr = self.train_epoch(train_loader, weights)
                vl = self.val_epoch(val_loader, weights)

                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']

                self._log_epoch(epoch, tr, vl, weights, lr)

                elapsed = time.time() - t0

                # Log to CSV
                writer.writerow({
                    'epoch': epoch, 'phase': phase_name,
                    'train_total': f"{tr['total']:.6f}",
                    'val_total':   f"{vl['total']:.6f}",
                    'data':  f"{tr.get('data',0):.6f}",
                    'kvl':   f"{tr.get('kvl',0):.6f}",
                    'kcl':   f"{tr.get('kcl',0):.6f}",
                    'dae':   f"{tr.get('dae',0):.6f}",
                    'bc':    f"{tr.get('bc',0):.6f}",
                    'ic':    f"{tr.get('ic',0):.6f}",
                    'pwr':   f"{tr.get('pwr',0):.6f}",
                    'lambda_kvl': f"{weights.get('kvl',0):.4f}",
                    'lambda_kcl': f"{weights.get('kcl',0):.4f}",
                    'lr': f"{lr:.2e}",
                    'time_s': f"{elapsed:.2f}",
                })
                f.flush()

                # Print every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"  Ep {epoch:4d} | {phase_name:30s} | "
                        f"Train: {tr['total']:.5f} | Val: {vl['total']:.5f} | "
                        f"KVL: {tr.get('kvl',0):.5f} | KCL: {tr.get('kcl',0):.5f} | "
                        f"LR: {lr:.2e}"
                    )

                # Best checkpoint (tracked on validation data loss to avoid curriculum weight shift anomalies)
                val_track = vl['data']
                if val_track < self.best_val_loss:
                    self.best_val_loss = val_track
                    self.no_improve    = 0
                    self.save_checkpoint(epoch, vl['total'], tag="best")
                else:
                    self.no_improve += 1

                # Early stopping
                if self.no_improve >= self.patience:
                    logger.info(f"  [Early Stop] No improvement for {self.patience} epochs.")
                    self.save_checkpoint(epoch, vl['total'], tag="last")
                    break

        # Save final
        self.save_checkpoint(self.epochs, self.best_val_loss, tag="final")
        logger.info(f"  [DONE] Training complete. Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"  Training log: {log_csv}")
        return self.history
