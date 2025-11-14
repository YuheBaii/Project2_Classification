"""
CIFAR-10 类别不平衡处理方案
实现多种处理不平衡数据的方法
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler

class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重，shape: [num_classes]
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]
            focal_loss = at * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(dataset, num_classes):
    """
    计算类别权重 - 方法1: 反比例权重
    """
    # 统计每个类别的样本数
    class_counts = np.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    # 计算权重（样本数的倒数）
    total = len(dataset)
    weights = total / (num_classes * class_counts)
    
    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {int(count)} samples, weight: {weights[i]:.4f}")
    
    return torch.FloatTensor(weights)

def compute_balanced_weights(dataset, num_classes):
    """
    计算平衡权重 - 方法2: sklearn风格
    """
    class_counts = np.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    # 使用sklearn的balanced权重计算方式
    n_samples = len(dataset)
    weights = n_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)

def create_weighted_sampler(dataset, num_classes):
    """
    创建加权采样器 - 用于过采样少数类
    """
    # 统计每个样本的权重
    class_counts = np.zeros(num_classes)
    targets = []
    
    for _, label in dataset:
        class_counts[label] += 1
        targets.append(label)
    
    # 计算每个样本的权重
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_imbalanced_cifar10(dataset, imbalance_ratios):
    """
    创建不平衡的CIFAR-10数据集
    
    Args:
        dataset: 原始CIFAR-10数据集
        imbalance_ratios: dict, {class_id: keep_ratio}
                         例如 {0: 1.0, 1: 0.5, 2: 0.1} 表示
                         类别0保留100%，类别1保留50%，类别2保留10%
    """
    from torch.utils.data import Subset
    
    indices = []
    for idx, (_, label) in enumerate(dataset):
        if label in imbalance_ratios:
            if np.random.rand() < imbalance_ratios[label]:
                indices.append(idx)
        else:
            indices.append(idx)
    
    return Subset(dataset, indices)

class ImbalanceCIFAR10DataModule:
    """
    不平衡CIFAR-10数据模块
    """
    def __init__(self, data_root, batch_size=64, num_workers=4,
                 imbalance_ratios=None, balance_method='none'):
        """
        Args:
            balance_method: 'none', 'class_weights', 'oversampling', 'focal_loss'
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imbalance_ratios = imbalance_ratios
        self.balance_method = balance_method
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def setup(self, train_tf=None, val_tf=None):
        import torchvision
        
        # 加载数据
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_tf
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=val_tf
        )
        
        # 创建不平衡数据集（如果指定）
        if self.imbalance_ratios:
            print(f"\nCreating imbalanced dataset with ratios: {self.imbalance_ratios}")
            train_dataset = create_imbalanced_cifar10(train_dataset, self.imbalance_ratios)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 根据平衡方法设置
        if self.balance_method == 'oversampling':
            self.sampler = create_weighted_sampler(train_dataset, len(self.class_names))
        else:
            self.sampler = None
            
        # 计算类别权重（用于loss）
        if self.balance_method in ['class_weights', 'focal_loss']:
            self.class_weights = compute_class_weights(train_dataset, len(self.class_names))
        else:
            self.class_weights = None
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# 使用示例配置
IMBALANCE_CONFIGS = {
    "severe": {
        # 严重不平衡：某些类别只有10%的数据
        0: 1.0, 1: 1.0, 2: 0.1, 3: 0.1, 4: 0.5,
        5: 0.5, 6: 0.3, 7: 0.3, 8: 0.2, 9: 0.2
    },
    "moderate": {
        # 中等不平衡：某些类别有50%的数据
        0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 0.7,
        5: 0.7, 6: 0.6, 7: 0.6, 8: 0.8, 9: 0.8
    },
    "mild": {
        # 轻度不平衡：最少的类别有80%的数据
        0: 1.0, 1: 1.0, 2: 0.8, 3: 0.8, 4: 0.9,
        5: 0.9, 6: 0.85, 7: 0.85, 8: 0.95, 9: 0.95
    }
}

if __name__ == "__main__":
    print("CIFAR-10 Imbalance Handling Methods:")
    print("1. Class Weights - 在loss中使用类别权重")
    print("2. Oversampling - 对少数类进行过采样")
    print("3. Focal Loss - 使用Focal Loss处理不平衡")
    print("\nUse these methods in your training config!")


