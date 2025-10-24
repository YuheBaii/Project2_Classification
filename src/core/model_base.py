import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class ModuleBase(nn.Module):  #@save
    """The base class of models."""
    def __init__(self):
        super().__init__()


    def compute_loss(self, y_hat : torch.Tensor, y: torch.Tensor, criterion=None) -> torch.Tensor:
        """compute_loss(y_hat, y) -> loss"""
        raise NotImplementedError

    def compute_metric(self, y_hat : torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """forward(X) -> y_hat"""
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """training_step(batch) -> loss"""
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        metrics = self.compute_metric(y_hat, y)
        return {'loss': loss, 'metrics': metrics}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """validation_step(batch) -> Dict of metrics"""
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        return {'val_loss': loss, 'metrics': self.compute_metric(y_hat, y)}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """test_step(batch) -> Dict of metrics"""
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        return {'loss': loss, 'metrics': self.compute_metric(y_hat, y)}

    # def optimizer(self) :
    #     """optimizer() -> optimizer"""
    #     raise NotImplementedError


def init_cnn(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if getattr(m, "weight", None) is not None and m.weight.dim() > 0:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)