import torch
import torch.nn as nn

# ---------------------------
# 3x3 and 1x1 conv helpers
# ---------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ---------------------------
# BasicBlock
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------
# ResNet for CIFAR10
# ---------------------------
class ResNet18Small(nn.Module):
    def __init__(self, num_classes=10, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # ---------------------------
    # For Trainer Compatibility
    # ---------------------------
    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        acc = self.compute_metric(y_hat, y)['accuracy']
        
        # loss 保持为 tensor (trainer会调用.backward()和.item())
        # accuracy 转为 Python 标量 (trainer会直接用于计算)
        return {
            'loss': loss,
            'metrics': {
                'accuracy': acc.item() if isinstance(acc, torch.Tensor) else acc
            }
        }


    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        acc = self.compute_metric(y_hat, y)['accuracy']
        
        # val_loss 也必须是 tensor (trainer会调用.item())
        # metrics 中的 key 必须是 "accuracy" (不是 "val_accuracy")
        return {
            'val_loss': loss,
            'metrics': {
                'accuracy': acc.item() if isinstance(acc, torch.Tensor) else acc
            }
        }


    def compute_loss(self, y_hat, y):
        criterion = self.loss_fn

        # 转 tensor
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=y_hat.device)
        else:
            y = y.to(y_hat.device)

        if y_hat.shape[1] == 1:  # Binary
            y = y.float()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            if y.ndim == 1:
                y = y[:, None]  # 变成 [batch,1]
        else:  # Multi-class
            y = y.long()
            if y.dim() == 0:
                y = y.unsqueeze(0)

        return criterion(y_hat, y)


    def compute_metric(self, y_hat, y):
        with torch.no_grad():
            # 转 tensor
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, device=y_hat.device)
            else:
                y = y.to(y_hat.device)

            if y_hat.shape[1] == 1:  # Binary
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                if y.ndim == 1:
                    y = y[:, None]
                pred = (torch.sigmoid(y_hat) > 0.5).long()
            else:  # Multi-class
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                pred = y_hat.argmax(dim=1)

            # 保证长度对齐
            if y_hat.shape[0] != pred.shape[0]:
                min_len = min(y_hat.shape[0], pred.shape[0])
                y = y[:min_len]
                pred = pred[:min_len]

            # flatten y 和 pred 到 1D
            acc = (pred.view(-1) == y.view(-1)).float().mean()
        
        # 返回字典格式
        return {'accuracy': acc}



