"""
CIFAR-10 类别不平衡处理方案
实现多种处理不平衡数据的方法
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import Subset

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
"""
class ImbalanceCIFAR10DataModule:
    
    #不平衡CIFAR-10数据模块
    
    #def __init__(self, data_root, batch_size=64, num_workers=4,
    #             imbalance_ratios=None, balance_method='none'):
    def __init__(self, data_root, batch_size=64, num_workers=4, 
                 imbalance_ratios=None, balance_method='none', 
                 #  关键修改：添加占位符以保存 transforms
                 train_tf=None, val_tf=None):
        
        #Args:
        #    balance_method: 'none', 'class_weights', 'oversampling', 'focal_loss'
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imbalance_ratios = imbalance_ratios
        self.balance_method = balance_method
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        # ⚠️ 关键修改：保存传入的 transforms
        self.train_tf = train_tf 
        self.val_tf = val_tf
        # 实例变量初始化
        self.train_dataset = None
        self.test_dataset = None

    #def setup(self, train_tf=None, val_tf=None):
    def setup(self, stage=None, train_tf=None, val_tf=None):
        import torchvision
        # 如果 setup 再次接收到 transforms，则更新
        if train_tf is not None:
            self.train_tf = train_tf
        if val_tf is not None:
            self.val_tf = val_tf
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
        
        # ⚠️ 关键修改：使用 self.val_tf 来创建测试集
        self.test_dataset = CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=self.val_tf  # <-- 必须使用保存的 self.val_tf
        )
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
    
    def test_dataloader(self):
        # 确保使用 setup 中创建的正确变换后的 test_dataset
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    """
# 路径: src/utils/imbalance.py

# ... (其他导入和辅助函数，如 create_imbalance_dataset)

def create_imbalance_dataset(dataset, imbalance_type, num_classes, balance_method):
    # 警告：这是简化版，用于避免 NameError。请确保您原有的逻辑在此处！
    
    # 假设您的 imbalance_ratios 在 IMBALANCE_CONFIGS["moderate"]
    if imbalance_type in IMBALANCE_CONFIGS:
        imbalance_ratios = IMBALANCE_CONFIGS[imbalance_type]
    else:
        imbalance_ratios = {i: 1.0 for i in range(num_classes)}
        
    indices = []
    # 简化：仅基于 ratios 随机抽取
    for idx, (_, label) in enumerate(dataset):
        if np.random.rand() < imbalance_ratios.get(label, 1.0):
            indices.append(idx)

    # 简化的结果返回
    subset = Subset(dataset, indices)
    # 模拟 class_weights 和 sampler 的返回值
    class_weights = torch.ones(num_classes)
    sampler = None
    
    # 打印创建信息 (可选)
    print(f"Creating imbalanced dataset with type: {imbalance_type} (Simplified stub used)")
    
    return subset, class_weights, sampler

# -------------------------------------------------------------------

# 替换整个 ImbalanceCIFAR10DataModule 类定义
class ImbalanceCIFAR10DataModule: # ⬅️ 不再继承 LightningDataModule
    def __init__(
        self,
        data_root: str,
        # ⚠️ 确保这些参数在您的 build_data 中被正确传入
        img_size: int, 
        num_classes: int,
        batch_size: int = 64,
        num_workers: int = 4,
        imbalance_type: str = "none",  
        balance_method: str = "none",  
        seed: int = 42,
    ):
        # 移除 super().__init__()
        self.data_root = data_root
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imbalance_type = imbalance_type
        self.balance_method = balance_method
        self.seed = seed
        
        # 属性初始化
        self.train_tf = None
        self.val_tf = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        self.sampler = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck'] # 添加 class_names
        
    # setup 方法签名不变，因为它被 build_data 调用
    def setup(self, stage=None, train_tf=None, val_tf=None):
        # 1. 保存/更新 transforms
        if train_tf is not None:
            self.train_tf = train_tf
        if val_tf is not None:
            self.val_tf = val_tf
        
        # 确保 val_tf 至少是 T.ToTensor()
        if self.val_tf is None:
            # 兼容性修复：如果 config 没提供 transforms，至少要能转 Tensor
            self.val_tf = T.ToTensor() 

        # 2. 加载原始 CIFAR-10 数据集 (使用传入或默认的 transforms)
        train_raw = CIFAR10(
            root=self.data_root, train=True, download=True, transform=self.train_tf
        )
        val_raw = CIFAR10(
            root=self.data_root, train=False, download=True, transform=self.val_tf
        )

        # 3. 创建不平衡训练集 (如果需要)
        if self.imbalance_type != "none" and self.imbalance_type is not None:
            # 使用项目中的 create_imbalance_dataset 函数
            train_dataset_imbalanced, self.class_weights, self.sampler = create_imbalance_dataset(
                dataset=train_raw, 
                imbalance_type=self.imbalance_type, 
                num_classes=self.num_classes, 
                balance_method=self.balance_method # ⚠️ 注意：如果你的 create_imbalance_dataset 没有这个参数需要调整
            )
            self.train_dataset = train_dataset_imbalanced
        else:
            self.train_dataset = train_raw

        self.val_dataset = val_raw # Val set 保持原始大小

        # 4. 创建 Test Dataset (核心修复点：确保 transform 包含 T.ToTensor())
        current_test_tf = self.val_tf
        safe_transforms = []

        if isinstance(current_test_tf, T.Compose):
            safe_transforms.extend(current_test_tf.transforms)
        elif current_test_tf is not None:
            safe_transforms.append(current_test_tf)
            
        has_to_tensor = any(isinstance(t, T.ToTensor) for t in safe_transforms)

        if not has_to_tensor:
            safe_transforms.append(T.ToTensor())
            print("⚠️ 警告: 强制在 Test Set Transforms 中加入 T.ToTensor()。")

        # 重新组合 safe_test_tf
        safe_test_tf = T.Compose(safe_transforms) if len(safe_transforms) > 1 else safe_transforms[0]
        
        self.test_dataset = CIFAR10( 
            root=self.data_root,
            train=False, 
            download=True,
            transform=safe_test_tf # ⬅️ 使用安全的变换
        )

    # 移除 train_dataloader, val_dataloader, test_dataloader 方法，因为 analyze_errors.py 只访问 test_dataset
    # 如果 build_data 还需要这些方法，您需要将其保留并确保它们返回 DataLoader
    def train_dataloader(self):
        # 检查是否使用 Sampler (例如，对于 oversampling 或 class_weights)
        if self.sampler is not None:
            shuffle = False # 使用 Sampler 时不能 shuffle
        else:
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self.sampler, # 应用 Sampler
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # 此时 self.test_dataset 已经使用了包含 T.ToTensor 的 safe_test_tf
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
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
