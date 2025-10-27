from torchvision import transforms as T
from torch import nn, optim
from src.datamodules.dogs_vs_cats import DogsVsCatsDataModule
from src.datamodules.cifar10 import CIFAR10DataModule
from src.models.cnn import SimpleCNNBinary
from src.losses.focal_loss import FocalLoss
from src.models.cnn_bn import BNCnn
from src.models.VGG import VGG

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
    """
    Build data transforms based on configuration.
    If cfg_aug is None or empty, returns None for both train and val transforms.
    """
    if cfg_aug is None:
        print("Warning: No augmentation config provided, using None transforms")
        return None, None
    
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
    elif t == "cifar10":
        dm = CIFAR10DataModule(
            data_root=cfg["task"]["data_root"],
            img_size=cfg["task"].get("img_size", 32),
            batch_size=bs,
            num_workers=nw,
            val_split=cfg["task"].get("val_split", 5000),
            train_tf=train_tf,
            val_tf=val_tf,
        )
        dm.setup()
        return dm, dm.class_names
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
    img_size = cfg["task"].get("img_size", 224)
    loss_fn = build_loss(cfg)
    loss_name = cfg["model"]["loss_fn"]
    print(f"Building model: {m} with loss: {loss_name}")
    if loss_name == "bce" and num_classes != 1:
        num_classes = 1  # For BCE, use single output neuron
    if m == "simple_cnn_binary":
        return SimpleCNNBinary(loss_fn=loss_fn, input_size=(img_size, img_size), num_classes=num_classes)
    
    elif m == "cnn_bn":
        return BNCnn(
            loss_fn=loss_fn, 
            input_size=(img_size, img_size), 
            num_classes=num_classes, 
            dropout_p=cfg["model"]["dropout_p"]
        )
    
    elif m == "vgg":
        # default to A，VGG-11
        arch_name = cfg["model"].get("arch_name", "A")
        vgg_cfgs = cfg["model"].get("vgg_cfgs", {})
        arch = vgg_cfgs.get(arch_name, [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])
        
        print(f"VGG arch ({arch_name}):", arch)
        print(f"VGG img_size: {img_size}")
        
        return VGG(arch=arch, loss_fn=loss_fn, img_size=img_size, num_classes=num_classes)
    
    elif m == "resnet18":
        from src.models.resnet import ResNet18
        arch = cfg["model"].get("arch", ((2, 64), (2, 128), (2, 256), (2, 512)))
        
        print(f"ResNet18 arch:", arch)
        print(f"ResNet18 img_size: {img_size}")
        
        return ResNet18(arch=arch, num_classes=num_classes, loss_fn=loss_fn)
    
    elif m == "resnext50":
        from src.models.resnext import ResNeXt
        # ResNeXt-50 32x4d: layers=(3,4,6,3), groups=32, width_per_group=4
        layers = cfg["model"].get("layers", (3,4,6,3))
        groups = cfg["model"].get("groups", 32)
        width_per_group = cfg["model"].get("width_per_group", 4)
        
        print(f"ResNeXt-50 layers: {layers}, groups: {groups}, width_per_group: {width_per_group}")
        print(f"ResNeXt-50 img_size: {img_size}")
        
        return ResNeXt(layers=layers, groups=groups, width_per_group=width_per_group,
                      num_classes=num_classes, loss_fn=loss_fn)

    
    raise ValueError(f"Unknown model: {m}")


def build_optimizer(cfg, model):
    ocfg = cfg.get("optim", {"name":"adamw", "lr":1e-3})
    params = [p for p in model.parameters() if p.requires_grad]
    if ocfg["name"].lower() == "sgd":
        return optim.SGD(params, lr=ocfg["lr"], momentum=ocfg.get("momentum", 0.9), weight_decay=ocfg.get("weight_decay", 0.0))
    else:
        return optim.AdamW(params, lr=ocfg.get("lr", 1e-3), weight_decay=ocfg.get("weight_decay", 0.01))
