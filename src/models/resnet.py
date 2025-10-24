# import torch.nn as nn
# from torchvision import models

# def build_resnet18(num_classes=2, pretrained=True, freeze_backbone=False):
#     m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
#     if freeze_backbone:
#         for p in m.parameters():
#             p.requires_grad = False
#     in_features = m.fc.in_features
#     m.fc = nn.Linear(in_features, num_classes)
#     return m
