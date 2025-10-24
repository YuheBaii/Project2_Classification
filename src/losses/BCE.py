import torch.nn as nn
def build_bce():
    return nn.BCEWithLogitsLoss()