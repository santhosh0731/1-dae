"""
Adaptive Learning - Online Learner
==================================
Handles in-memory neural surrogate fine-tuning on copied model weights.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

class OnlineLearner:
    """Manages online fine-tuning on cloned model checkpoints."""
    
    def __init__(self, base_model: nn.Module, lr: float = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        # Clone model weights to prevent modifying the base model files
        self.adapted_model = copy.deepcopy(base_model).to(self.device)
        self.optimizer = optim.SGD(self.adapted_model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train_step(self, X_batch, Y_batch):
        """Runs a single gradient descent update step on the copied weights."""
        self.adapted_model.train()
        self.optimizer.zero_grad()
        
        # Move tensors to device as float32
        inputs = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
        targets = torch.tensor(Y_batch, dtype=torch.float32, device=self.device)
        
        preds = self.adapted_model(inputs)
        if isinstance(preds, dict):
            preds = preds['preds']
            
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optimizer.step()
        
        self.adapted_model.eval()
        return float(loss.item())

    def get_model(self) -> nn.Module:
        return self.adapted_model
