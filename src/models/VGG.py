from __future__ import annotations
import torch
from torch import nn
from src.core.model_base import ModuleBase

class _Rescale01(nn.Module):
    """Scale raw [0,255] images to [0,1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG(ModuleBase):
    def __init__(self, arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), loss_fn=None, img_size=224, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        
        # 根据 num_classes 决定输出维度
        # 二分类: num_classes=2 用 CrossEntropy 或 num_classes=1 用 BCE
        out_features = num_classes if num_classes > 1 else 1
        
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(out_features))
        self.rescale = _Rescale01()
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        
        # Initialize lazy layers with a dummy forward pass, then apply custom weights
        # This must be done outside of the actual forward to avoid inplace modification during training
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)  # Use img_size from config
            _ = self.net(self.rescale(dummy_input))
        
        # Now apply custom initialization after lazy layers are materialized
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("input:", x.shape)
        x = self.rescale(x)          # [B,3,H,W] -> scaled
        #print("rescaled:", x.shape)
        x = self.net(x)         # conv stacks    
        return x

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