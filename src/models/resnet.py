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

class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
class ResNet18(ModuleBase):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)
    
    def __init__(self, arch=((2, 64), (2, 128), (2, 256), (2, 512)), num_classes=2, loss_fn=None):
        super(ResNet18, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        
        out_features = num_classes if num_classes > 1 else 1
        # Build network
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
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
        # x = self.rescale(x)  # 输入已经经过 ToTensor() 和 Normalize() 处理
        return self.net(x)
    
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, criterion=None) -> torch.Tensor:
        """Compute loss based on output dimension and loss function."""
        criterion = criterion if criterion is not None else self.loss_fn
        
        if y_hat.shape[1] == 1:  # Binary classification with BCE
            #y = y.float().view(-1, 1)
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

