# src/datamodules/dogs_vs_cats_dm.py
from __future__ import annotations
import os, glob
from typing import Optional, Sequence
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from src.core.datamodule_base import DataModuleBase

class _FolderNoLabel(Dataset):
    def __init__(self, files: Sequence[str], transform=None):
        self.files = list(files)
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, os.path.basename(p)


class DogsVsCatsDataModule(DataModuleBase):
    def __init__(
        self,
        data_root: str,
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        train_tf = None,
        val_tf = None,
        **kwargs,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, **kwargs)
        self.data_root = data_root
        self.img_size = img_size
        self.train_dir = os.path.join(data_root, train_dir)
        self.val_dir   = os.path.join(data_root, val_dir)
        self.test_dir  = os.path.join(data_root, test_dir)
        self.class_names = None
        self.persistent_workers = True
        self.train_tf = train_tf
        self.val_tf = val_tf


    def setup(self, stage: Optional[str] = None):
        size = (180, 180)
        if self.train_tf is None:
            self.train_tf = T.Compose([T.Resize(size), T.ToTensor()])
        if self.val_tf is None:
            self.val_tf   = T.Compose([T.Resize(size), T.ToTensor()])
        # 训练/验证
        if stage in (None, "fit", "validate"):

            self.train_ds = ImageFolder(self.train_dir, transform=self.train_tf)
            self.val_ds   = ImageFolder(self.val_dir,   transform=self.val_tf)

            self.class_names = self.train_ds.classes

        # 测试（无标签）
        if stage in (None, "test"):
            test_files = sorted(glob.glob(os.path.join(self.test_dir, "*")))
            self.test_ds = _FolderNoLabel(test_files, transform=self.val_tf)


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
