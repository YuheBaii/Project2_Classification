# src/models/cnn_binary.py
from __future__ import annotations
import torch
from torch import nn
from src.core.model_base import ModuleBase
from torch.nn.parameter import UninitializedParameter

def _init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
        # 跳过尚未物化的 Lazy 参数
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

class _Rescale01(nn.Module):
    """Scale raw [0,255] images to [0,1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

class BNCnn(ModuleBase):
    """
    CNN for binary classification (dog vs cat), Keras-style:
    Input -> Rescale -> [Conv+Pool]*4 -> Conv -> Flatten -> Dense
    - Convs: kernel=3, stride=1, padding=0 (Keras default 'valid')
    - Pools: MaxPool2d(2)
    - Loss: Supports both BCEWithLogitsLoss and CrossEntropyLoss
    """
    def __init__(self, in_channels: int = 3, input_size: tuple[int,int]=(180,180), loss_fn=None, num_classes=2, dropout_p=0.4):
        super().__init__()
        self.num_classes = num_classes
        H, W = input_size

        # Feature extractors
        self.rescale = _Rescale01()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0,bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0,bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0,bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 5 (no pooling)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0,bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        # 二分类: num_classes=2 用 CrossEntropy 或 num_classes=1 用 BCE
        out_features = num_classes if num_classes > 1 else 1
        self.classifier = nn.Sequential(
            nn.Flatten(),nn.Dropout(p=dropout_p),
            nn.Linear(256, out_features)
        )

        # Loss function (default to BCEWithLogitsLoss if not provided)
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

        # Init
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        print("input:", x.shape)
        x = self.rescale(x)          # [B,3,H,W] -> scaled
        print("rescaled:", x.shape)
        x = self.features(x)         # conv stacks    
        print("features:", x.shape)
        logit = self.classifier(x)   # [B,1] (logits)
        print("logit:", logit.shape)
        return logit

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, criterion=None) -> torch.Tensor:
        # Use provided criterion, otherwise use the loss function set during initialization
        criterion = criterion if criterion is not None else self.loss_fn
        print(f"DEBUG: Using criterion: {criterion}")
        
        if y_hat.shape[1] == 1: 
            y = y.float().view(-1, 1)
        else:  
            y = y.long()
        return criterion(y_hat, y)


    def compute_metric(self, y_hat: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            if y_hat.shape[1] == 1:  
                probs = torch.sigmoid(y_hat)
                preds = (probs >= 0.5).long().view(-1)
            else:  
                preds = torch.argmax(y_hat, dim=1)
            
            acc = (preds == y.view(-1).long()).float().mean().item()
        return {"accuracy": acc}
