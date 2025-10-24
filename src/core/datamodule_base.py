from abc import ABC, abstractmethod
from typing import Optional

class DataModuleBase(ABC):  #@save
    """The base class of data."""
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def train_dataloader(self, batch_size):
        pass

    @abstractmethod
    def test_dataloader(self, batch_size):
        pass

    @abstractmethod
    def val_dataloader(self, batch_size):
        pass

