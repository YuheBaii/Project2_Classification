import torch
from torch import nn
from .core.model_base import ModuleBase, init_cnn

class LeNet(ModuleBase):
    def __init__(self, num_classes=10, lr=0.1):
        super().__init__()  
        self.num_classes = num_classes
        self.lr = lr
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(self.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.net.apply(init_cnn)
    
    def forward(self, X):
        """forward(X) -> y_hat"""
        return self.net(X)
    
    def compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def compute_metric(self, y_hat, y):
        with torch.no_grad():
            preds = torch.argmax(y_hat, dim=1)
            accuracy = (preds == y).float().mean()
        return {'accuracy': accuracy}

    def optimizer(self):
        return self._optimizer
    
    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)  
        for layer in self.net:
            X = layer(X) 
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
