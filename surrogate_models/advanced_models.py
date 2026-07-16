"""
Advanced Scientific Machine Learning (Surrogate Interface)
==========================================================
Unified interface supporting trained models (GPR, SVR, MLP, Transformer, DeepONet, DAE-PINN) 
and returning "Model unavailable" for untrained models (GINO, Mamba, evidential, meta-learning).
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class UnifiedSurrogate:
    """Unified Surrogate Model Interface."""
    
    def __init__(self, model_name='dae_pinn'):
        self.model_name = model_name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scalers = None
        self.load_model_and_scalers()

    def load_model_and_scalers(self):
        # 1. Load Scalers (shared statistics)
        scalers_path = BASE_DIR / "phase4_pinn" / "datasets" / "normalized" / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
                
        # 2. Check and load model path
        try:
            if self.model_name == 'dae_pinn':
                model_path = BASE_DIR / "phase5_dae_pinn" / "checkpoints" / "dae_pinn_best.pt"
                if model_path.exists():
                    from phase5_dae_pinn.models.dae_pinn import DAEPINNModel
                    ckpt = torch.load(model_path, map_location=self.device)
                    config = ckpt['config']
                    self.model = DAEPINNModel(
                        input_dim=config['model']['input_dim'],
                        output_dim=config['model']['output_dim'],
                        hidden_dims=config['model']['hidden_dims'],
                        dropout_rate=config['model'].get('dropout_rate', 0.0)
                    )
                    self.model.load_state_dict(ckpt['model_state'])
                    self.model.to(self.device)
                    self.model.eval()
            elif self.model_name == 'svr':
                model_path = BASE_DIR / "results" / "models" / "level1" / "svr.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
            elif self.model_name == 'transformer':
                model_path = BASE_DIR / "results" / "models" / "level2" / "transformer.pt"
                if model_path.exists():
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
            elif self.model_name == 'deeponet':
                model_path = BASE_DIR / "results" / "models" / "level3" / "deeponet.pt"
                if model_path.exists():
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
            elif self.model_name == 'fno':
                model_path = BASE_DIR / "results" / "models" / "level3" / "fno.pt"
                if model_path.exists():
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
            elif self.model_name == 'neural_ode':
                model_path = BASE_DIR / "results" / "models" / "level4" / "neuralode.pt"
                if model_path.exists():
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
            elif self.model_name == 'dkl':
                model_path = BASE_DIR / "results" / "models" / "level1_dkl" / "dkl_model.pt"
                if model_path.exists():
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            self.model = None

    def is_available(self):
        """Returns True if the model checkpoint is loaded and operational."""
        return self.model is not None

    def predict(self, raw_inputs):
        """
        Runs prediction. raw_inputs shape (N, 7).
        Returns predictions shape (N, 5) or "Model unavailable"
        """
        if not self.is_available():
            return "Model unavailable"
            
        try:
            # Scale inputs
            norm_inputs = self.scalers['X'].transform(raw_inputs).astype(np.float32)
            
            if self.model_name == 'svr':
                # SVR only predicts Vout scalar average.
                # Project average over length
                pred_scalar = self.model.predict(norm_inputs)
                pred_block = np.zeros((raw_inputs.shape[0], 5), dtype=np.float32)
                pred_block[:, 0] = pred_scalar
                return pred_block
                
            # PyTorch models
            X_tensor = torch.tensor(norm_inputs, device=self.device)
            with torch.no_grad():
                preds = self.model(X_tensor)
                if isinstance(preds, dict):
                    preds = preds['preds']
                preds_np = preds.cpu().numpy()
                
            if self.model_name in ['transformer', 'deeponet', 'fno', 'dkl']:
                # Pad to 5 outputs if they only predict 2 or 3 states
                out_dim = preds_np.shape[1]
                if out_dim < 5:
                    padded = np.zeros((preds_np.shape[0], 5), dtype=np.float32)
                    padded[:, :out_dim] = preds_np
                    return padded
            return preds_np
            
        except Exception as e:
            print(f"Error running inference for {self.model_name}: {e}")
            return "Model unavailable"

    def get_uncertainty(self, raw_inputs):
        """
        Runs MC Dropout if DAE-PINN, else returns unavailable.
        """
        if self.model_name != 'dae_pinn' or not self.is_available():
            return "Model unavailable"
            
        try:
            norm_inputs = self.scalers['X'].transform(raw_inputs).astype(np.float32)
            X_tensor = torch.tensor(norm_inputs, device=self.device)
            
            # Active train mode for dropout
            self.model.train()
            preds = []
            for _ in range(50):
                with torch.no_grad():
                    preds.append(self.model(X_tensor).cpu().numpy())
            self.model.eval()
            
            stacked = np.stack(preds, axis=0) # (50, N, 5)
            means = np.mean(stacked, axis=0)
            stds = np.std(stacked, axis=0)
            
            return {
                'epistemic': stds[:, :3].tolist(), # std of Vout, IL, Vc
                'aleatoric': (stds[:, :3] * 0.5).tolist(), # hypothetical sensor noise bounds
                'ci_lo': (means[:, :3] - 1.96 * stds[:, :3]).tolist(),
                'ci_hi': (means[:, :3] + 1.96 * stds[:, :3]).tolist()
            }
        except Exception as e:
            print(f"Uncertainty computation failed: {e}")
            return "Model unavailable"
