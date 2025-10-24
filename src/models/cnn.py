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

# def _init_weights(m: nn.Module):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)

class SimpleCNNBinary(ModuleBase):
    """
    CNN for binary classification (dog vs cat), Keras-style:
    Input -> Rescale -> [Conv+Pool]*4 -> Conv -> Flatten -> Dense
    - Convs: kernel=3, stride=1, padding=0 (Keras default 'valid')
    - Pools: MaxPool2d(2)
    - Loss: Supports both BCEWithLogitsLoss and CrossEntropyLoss
    """
    def __init__(self, in_channels: int = 3, input_size: tuple[int,int]=(180,180), loss_fn=None, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        H, W = input_size

        # Feature extractors
        self.rescale = _Rescale01()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 5 (no pooling)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
        )

        # Infer flatten dim with a dry run (CPU safe)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            feat = self.features(self.rescale(dummy))
            self._flat_dim = feat.view(1, -1).size(1)

        # 根据 num_classes 决定输出维度
        # 二分类: num_classes=2 用 CrossEntropy 或 num_classes=1 用 BCE
        out_features = num_classes if num_classes > 1 else 1
        self.classifier = nn.Linear(self._flat_dim, out_features)

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
        x = torch.flatten(x, 1)
        print("flattened:", x.shape)
        logit = self.classifier(x)   # [B,1] (logits)
        print("logit:", logit.shape)
        return logit

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, criterion=None) -> torch.Tensor:
        # Use provided criterion, otherwise use the loss function set during initialization
        criterion = criterion if criterion is not None else self.loss_fn
        print(f"DEBUG: Using criterion: {criterion}")
        
        # 根据输出维度判断是BCE还是CE
        if y_hat.shape[1] == 1:  # BCE: 输出是 (batch, 1)
            y = y.float().view(-1, 1)
        else:  # CE: 输出是 (batch, num_classes), 标签是类别索引
            y = y.long()
        
        return criterion(y_hat, y)


    def compute_metric(self, y_hat: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            # 根据输出维度判断预测方式
            if y_hat.shape[1] == 1:  # BCE: 使用sigmoid + 阈值
                probs = torch.sigmoid(y_hat)
                preds = (probs >= 0.5).long().view(-1)
            else:  # CE: 使用argmax
                preds = torch.argmax(y_hat, dim=1)
            
            acc = (preds == y.view(-1).long()).float().mean().item()
        return {"accuracy": acc}
