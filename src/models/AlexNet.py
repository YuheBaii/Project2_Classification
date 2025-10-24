import torch
from torch import nn
from src.core.model_base import ModuleBase, init_cnn

class AlexNet(ModuleBase):
    def __init__(self, num_classes=10,dropout_p: float = 0.5):
        super().__init__()  
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(init_cnn)
    
    def forward(self, X):
        """forward(X) -> y_hat"""
        X = self.features(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X

    def compute_loss(self, y_hat, y, criterion=None):
        crit = criterion if criterion is not None else self.loss_fn
        return crit(y_hat, y)

    def compute_metric(self, y_hat: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            acc = (y_hat.argmax(dim=1) == y).float().mean().item()
        return {"accuracy": acc}

    