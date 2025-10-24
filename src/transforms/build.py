from torchvision import transforms as T
from torch import nn, optim
from src.datamodules.dogs_vs_cats import DogsVsCatsDataModule
from src.datamodules.cifar10 import build_cifar10
from src.models.resnet import build_resnet18
from src.models.cnn import SimpleCNNBinary
from src.losses.focal_loss import FocalLoss


def _op(op):
    t = op["type"]
    if t == "RandomResizedCrop":
        return T.RandomResizedCrop(size=op["size"], scale=tuple(op.get("scale", (0.8,1.0))))
    if t == "RandomHorizontalFlip":
        return T.RandomHorizontalFlip(p=op.get("p", 0.5))
    if t == "Resize":
        return T.Resize(op["size"])
    if t == "CenterCrop":
        return T.CenterCrop(op["size"])
    if t == "ToTensor":
        return T.ToTensor()
    if t == "Normalize":
        return T.Normalize(mean=op["mean"], std=op["std"])
    raise ValueError(f"Unknown transform: {t}")

def build_transforms(cfg_aug):
    train_ops = cfg_aug.get("train", [])
    val_ops = cfg_aug.get("val", [])
    train_tf = T.Compose([_op(x) for x in train_ops]) if train_ops else None
    val_tf = T.Compose([_op(x) for x in val_ops]) if val_ops else None
    return train_tf, val_tf


def build_loss(cfg):
    loss_name = cfg["model"]["loss_fn"]
    if loss_name == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0))
    elif loss_name == "bce":
        from src.losses.BCE import build_bce
        return build_bce()
    else:
        from src.losses.CE import build_ce
        return build_ce()

def build_data(cfg, train_tf=None, val_tf=None):
    t = cfg["task"]["name"]
    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 4)
    if t == "dogs_vs_cats":
        dm = DogsVsCatsDataModule(
            data_root=cfg["task"]["data_root"],
            img_size=cfg["task"].get("img_size", 224),
            batch_size=bs,
            num_workers=nw,
            train_dir=cfg["task"].get("train_dir", "train"),
            val_dir=cfg["task"].get("val_dir", "val"),
            test_dir=cfg["task"].get("test_dir", "test"),
            train_tf=train_tf,
            val_tf=val_tf,
        )
        dm.setup()
        #return dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader(), dm.class_names
        return dm,dm.class_names
    # elif t == "cifar10":
    #     return build_cifar10(cfg["task"], train_tf, val_tf, bs, nw)
    else:
        raise ValueError(f"Unknown task: {t}")

def build_trainer(cfg):
    from src.core.trainer_base import TrainerBase
    return TrainerBase(
        epochs=cfg.get("epochs", 5),
        device=cfg.get("device", "cpu"),
    )
def build_model(cfg, num_classes):
    m = cfg["model"]["name"]
    if m == "simple_cnn_binary":
        # Build loss function based on config and pass it to the model
        img_size = cfg["task"].get("img_size", 224)
        loss_fn = build_loss(cfg)
        loss_name = cfg["model"]["loss_fn"]
        cnn_num_classes = 2 if loss_name == "ce" else 1
        
        print(f"SimpleCNN img_size: {img_size}")
        print(f"SimpleCNN num_classes: {cnn_num_classes} (loss: {loss_name})")
        
        return SimpleCNNBinary(loss_fn=loss_fn, input_size=(img_size, img_size), num_classes=cnn_num_classes)
    if m == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
            freeze_backbone=cfg["model"].get("freeze_backbone", False),
        )
    if m == "vgg":
        from src.models.VGG import VGG
        # 架构默认为 A，VGG-11
        arch_name = cfg["model"].get("arch_name", "A")
        vgg_cfgs = cfg["model"].get("vgg_cfgs", {})
        arch = vgg_cfgs.get(arch_name, [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])
        img_size = cfg["task"].get("img_size", 224)
        loss_fn = build_loss(cfg)

        loss_name = cfg["model"]["loss_fn"]
        vgg_num_classes = 2 if loss_name == "ce" else 1
        
        print(f"VGG arch ({arch_name}):", arch)
        print(f"VGG img_size: {img_size}")
        print(f"VGG num_classes: {vgg_num_classes} (loss: {loss_name})")
        
        return VGG(arch=arch, loss_fn=loss_fn, img_size=img_size, num_classes=vgg_num_classes)
    raise ValueError(f"Unknown model: {m}")


def build_optimizer(cfg, model):
    ocfg = cfg.get("optim", {"name":"adamw", "lr":1e-3})
    params = [p for p in model.parameters() if p.requires_grad]
    if ocfg["name"].lower() == "sgd":
        return optim.SGD(params, lr=ocfg["lr"], momentum=ocfg.get("momentum", 0.9), weight_decay=ocfg.get("weight_decay", 0.0))
    else:
        return optim.AdamW(params, lr=ocfg.get("lr", 1e-3), weight_decay=ocfg.get("weight_decay", 0.01))
