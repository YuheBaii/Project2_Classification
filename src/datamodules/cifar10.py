from typing import Optional
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
from src.core.datamodule_base import DataModuleBase

class CIFAR10DataModule(DataModuleBase):
    def __init__(
        self,
        data_root: str,
        img_size: int = 32,
        batch_size: int = 64,
        num_workers: int = 4,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        val_split: int = 5000,
        train_tf=None,
        val_tf=None,
        **kwargs,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, **kwargs)
        self.data_root = data_root
        self.img_size = img_size
        self.val_split = val_split
        self.class_names = None
        self.persistent_workers = True
        self.train_tf = train_tf
        self.val_tf = val_tf

    def setup(self, stage: Optional[str] = None):
        size = (self.img_size, self.img_size)
        if self.train_tf is None:
            self.train_tf = T.Compose([T.Resize(size), T.ToTensor()])
        if self.val_tf is None:
            self.val_tf   = T.Compose([T.Resize(size), T.ToTensor()])

        # 训练/验证
        if stage in (None, "fit", "validate"):
            full_train = CIFAR10(root=self.data_root, train=True, download=True, transform=self.train_tf)
            if self.val_split > 0:
                train_len = len(full_train) - self.val_split
                self.train_ds, self.val_ds = random_split(full_train, [train_len, self.val_split])
            else:
                self.train_ds = full_train
                self.val_ds = None
            self.class_names = full_train.classes

        # 测试（CIFAR10 标准测试集有标签，但项目推理使用无标签文件夹，所以这里不实现）
        if stage in (None, "test"):
            # 如需使用 CIFAR10 官方测试集，可以取消注释：
            # self.test_ds = CIFAR10(root=self.data_root, train=False, download=True, transform=self.val_tf)
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return None