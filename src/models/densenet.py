from __future__ import annotations
import torch
from torch import nn
from src.core.model_base import ModuleBase
from torch.nn.parameter import UninitializedParameter
from torch.nn import functional as F

def _init_weights(m: nn.Module):
    """Initialize weights for Conv2d, Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
        if isinstance(getattr(m, "weight", None), UninitializedParameter):
            return
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.LazyLinear)):
        if isinstance(getattr(m, "weight", None), UninitializedParameter):
            return
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
    
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(ModuleBase):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4), num_classes=2, loss_fn=None):
        super().__init__()
        self.num_channels = num_channels
        self.growth_rate = growth_rate
        self.arch = arch
        self.num_classes = num_classes
        
        out_features = num_classes if num_classes > 1 else 1
        
        # Build network
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                            growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(out_features)))
        
        # Loss function (default to appropriate loss based on num_classes)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            # 二分类默认 BCEWithLogitsLoss, 多分类默认 CrossEntropyLoss
            self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        
        # Initialize weights
        self.apply(_init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, criterion=None) -> torch.Tensor:
        """Compute loss based on output dimension and loss function."""
        criterion = criterion if criterion is not None else self.loss_fn
        
        if y_hat.shape[1] == 1:  # Binary classification with BCE
            y = y.float().view(y_hat.shape)
        else:  # Multi-class with CE
            y = y.long().view(-1)
        
        return criterion(y_hat, y)
    
    def compute_metric(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Compute accuracy metric."""
        with torch.no_grad():
            if y_hat.shape[1] == 1:  # Binary classification
                pred = (torch.sigmoid(y_hat) > 0.5).long().view(-1)
                y = y.view(-1)
            else:  # Multi-class classification
                pred = y_hat.argmax(dim=1)
                y = y.view(-1)
            
            acc = (pred == y).float().mean().item()
        return {'accuracy': acc}