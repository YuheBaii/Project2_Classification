# import os
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10

# def build_cifar10(cfg_task, train_tf, val_tf, batch_size, num_workers):
#     root = cfg_task["data_root"]
#     train_ds = CIFAR10(root=root, train=True, download=True, transform=train_tf)
#     val_ds = CIFAR10(root=root, train=False, download=True, transform=val_tf)
#     class_names = cfg_task["class_names"]
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     # reuse val split as test for demo
#     test_loader = val_loader
#     return train_loader, val_loader, test_loader, class_names
