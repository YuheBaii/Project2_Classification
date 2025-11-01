from __future__ import annotations
import torch
from torch import nn
from src.core.model_base import ModuleBase

def _init_weights(m: nn.Module):
    """Initialize weights for Conv2d, Linear layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ResNeXtBottleneck(nn.Module):
    expansion = 4  # 输出通道扩展倍数

    def __init__(self, inplanes, planes, stride=1, groups=32, width_per_group=4, downsample=None):
        super().__init__()
        # 瓶颈中间宽度：C×w
        width = int(groups * width_per_group)

        # 1x1 降维到 width
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        # 3x3 分组卷积（groups=C），stride 可为 1/2 下采样
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(width)

        # 1x1 升维回 planes*expansion
        out_channels = planes * self.expansion
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 1x1 shortcut 对齐形状（当 stride=2 或 inplanes!=out）

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)  # 1x1 conv + BN（stride 同步）

        out += identity
        return self.relu(out)

def make_layer(inplanes, planes, blocks, stride, groups, width_per_group):
    downsample = None
    out_channels = planes * ResNeXtBottleneck.expansion
    if stride != 1 or inplanes != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    layers = [ResNeXtBottleneck(inplanes, planes, stride,
                                groups, width_per_group, downsample)]
    inplanes = out_channels
    for _ in range(1, blocks):
        layers.append(ResNeXtBottleneck(inplanes, planes, 1, groups, width_per_group))
    return nn.Sequential(*layers), inplanes

class ResNeXt(ModuleBase):
    def __init__(self, layers=(3,4,6,3), groups=32, width_per_group=4, num_classes=2, loss_fn=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Network architecture
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        inplanes = 64
        self.layer1, inplanes = make_layer(inplanes,  64, layers[0], 1, groups, width_per_group)
        self.layer2, inplanes = make_layer(inplanes, 128, layers[1], 2, groups, width_per_group)
        self.layer3, inplanes = make_layer(inplanes, 256, layers[2], 2, groups, width_per_group)
        self.layer4, inplanes = make_layer(inplanes, 512, layers[3], 2, groups, width_per_group)

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        out_features = num_classes if num_classes > 1 else 1
        self.fc = nn.Linear(512*ResNeXtBottleneck.expansion, out_features)
        
        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.rescale(x)  # 输入已经经过 ToTensor() 和 Normalize() 处理
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
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


