from torchvision import transforms as T
from torch import nn, optim
from src.datamodules.dogs_vs_cats import DogsVsCatsDataModule
from src.datamodules.cifar10 import CIFAR10DataModule
from src.models.cnn import SimpleCNNBinary
from src.losses.focal_loss import FocalLoss
from src.models.cnn_bn import BNCnn
from src.models.VGG import VGG
from src.utils.imbalance import ImbalanceCIFAR10DataModule, IMBALANCE_CONFIGS

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


#def build_loss(cfg):
#def build_loss(cfg, class_weights=None): # <--- 🚨 关键修复点：添加 class_weights 参数
#    # 若任务包含不平衡CIFAR-10
#   if "task" in cfg and cfg["task"]["name"] == "cifar10_imbalanced":
#        balance_method = cfg["task"].get("balance_method", None)
#        if balance_method == "class_weights":
#            from src.utils.imbalance import ImbalanceCIFAR10DataModule, IMBALANCE_CONFIGS, compute_class_weights
#            import torchvision
#            from torch import nn

#            tmp_dataset = torchvision.datasets.CIFAR10(root=cfg["task"]["data_root"], train=True, download=True)
#            weights = compute_class_weights(tmp_dataset, num_classes=10)
#            print(f"Using class-weighted CE loss. Weights = {weights.tolist()}")
#            return nn.CrossEntropyLoss(weight=weights)

#    loss_name = cfg["model"]["loss_fn"]
#    if loss_name == "focal":
#        return FocalLoss(gamma=cfg.get("gamma", 2.0))
#    elif loss_name == "bce":
#        from src.losses.BCE import build_bce
#        return build_bce()
#    else:
#        from src.losses.CE import build_ce
#        return build_ce()


def build_loss(cfg, class_weights=None): # 确保签名是 build_loss(cfg, class_weights=None)
    from torch import nn
    loss_name = cfg["model"]["loss_fn"]
    
    # 1. 检查是否需要使用类别权重（来自 DataModule）
    if "task" in cfg and cfg["task"]["name"] == "cifar10_imbalanced":
        balance_method = cfg["task"].get("balance_method", None)
        
        # 使用 class_weights 进行 CE loss
        if balance_method == "class_weights":
            if class_weights is not None:
                print(f"Using class-weighted CE loss. Weights = {class_weights.tolist()}")
                # 传入 DataModule 已经计算好的不平衡权重
                return nn.CrossEntropyLoss(weight=class_weights)
            else:
                print("Warning: class_weights requested but not provided. Using standard CE loss.")

        # 使用 Focal Loss
        elif balance_method == "focal_loss":
            # Focal Loss 通常也使用权重（alpha）来处理不平衡
            gamma = cfg.get("gamma", 2.0)
            if class_weights is not None:
                print(f"Using Focal Loss with alpha weights (gamma={gamma}).")
            else:
                print(f"Using standard Focal Loss (gamma={gamma}).")
            # 传入 class_weights 作为 alpha
            return FocalLoss(alpha=class_weights, gamma=gamma)
            
        # 如果是 imbalanced 任务但没选平衡方法，或者权重没传进来，退回到标准 CE
        
    # 2. 默认损失函数逻辑
    
    if loss_name == "focal":
        # 这是一个兼容性回退，如果配置了 focal 但没有走 imbalanced 逻辑，则使用标准 focal
        return FocalLoss(gamma=cfg.get("gamma", 2.0))
        
    elif loss_name == "bce":
        from src.losses.BCE import build_bce
        return build_bce()
    
    else: # 默认 CE
        from src.losses.CE import build_ce
        # 最后的保障：如果 balance_method 是 class_weights 但权重没传进来，这里会用无权重的 CE
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

    elif t == "cifar10_imbalanced":
        # 使用 ImbalanceCIFAR10DataModule
        from src.utils.imbalance import ImbalanceCIFAR10DataModule, IMBALANCE_CONFIGS
        
        task_cfg = cfg["task"]
        
        imbalance_type = task_cfg.get("imbalance_type", "moderate")
        balance_method = task_cfg.get("balance_method", "class_weights")
        
        dm = ImbalanceCIFAR10DataModule(
            data_root=task_cfg["data_root"],
            img_size=task_cfg.get("img_size", 32),
            num_classes=task_cfg.get("num_classes", 10),
            batch_size=bs,
            num_workers=nw,
            imbalance_type=imbalance_type,
            balance_method=balance_method
        )
        
        # setup 会创建不平衡数据集并计算 class_weights
        dm.setup(train_tf=train_tf, val_tf=val_tf)
        
        # 🚨 关键修复点 A：返回 datamodule 和 class_weights
        return dm, dm.class_names, dm.class_weights # <--- 添加 dm.class_weights
    
    else:
        raise ValueError(f"Unknown task: {t}")

def build_trainer(cfg, run_dir=None):
    from src.core.trainer_base import TrainerBase
    return TrainerBase(
        epochs=cfg.get("epochs", 5),
        device=cfg.get("device", "cpu"),
        patience=cfg.get("early_stopping", {}).get("patience", 5),
        min_delta=cfg.get("early_stopping", {}).get("min_delta", 0.0),
        run_dir=run_dir,
    )
#def build_model(cfg, num_classes):
def build_model(cfg, num_classes, class_weights=None): # <--- 添加 class_weights 参数
    m = cfg["model"]["name"]
    img_size = cfg["task"].get("img_size", 224)
    #loss_fn = build_loss(cfg)
    loss_fn = build_loss(cfg, class_weights=class_weights)
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
        arch_list = vgg_cfgs.get(arch_name, [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])
        
        # 将列表转换为元组格式 (VGG模型期望元组的元组)
        arch = tuple(tuple(block) for block in arch_list)
        
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
    
    #elif m == "resnet18_small":
    #    from src.models.resnet_small import ResNet18Small
    #    arch = cfg["model"].get("arch", ((2, 32), (2, 64), (2, 128), (2, 256)))
    #    
    #    print(f"ResNet18 arch:", arch)
    #    print(f"ResNet18 img_size: {img_size}")
    #    
    #    return ResNet18Small(arch=arch, num_classes=num_classes, loss_fn=loss_fn)

    elif m == "resnet18_small":
        from src.models.resnet_small import ResNet18Small
        print(f"Building CIFAR10-style ResNet18 (img_size={img_size}, num_classes={num_classes})")
        return ResNet18Small(num_classes=num_classes, loss_fn=loss_fn)


    elif m == "densenet":
        from src.models.densenet import DenseNet
        # DenseNet-121: arch=(6,12,24,16), growth_rate=32
        num_channels = cfg["model"].get("num_channels", 64)
        growth_rate = cfg["model"].get("growth_rate", 32)
        arch = cfg["model"].get("arch", (4, 4, 4, 4))
        
        print(f"DenseNet arch: {arch}, growth_rate: {growth_rate}, num_channels: {num_channels}")
        print(f"DenseNet img_size: {img_size}")
        
        return DenseNet(num_channels=num_channels, growth_rate=growth_rate, arch=arch,
                       num_classes=num_classes, loss_fn=loss_fn)
    
    raise ValueError(f"Unknown model: {m}")


def build_optimizer(cfg, model):
    ocfg = cfg.get("optim", {"name":"adamw", "lr":1e-3})
    params = [p for p in model.parameters() if p.requires_grad]
    if ocfg["name"].lower() == "sgd":
        return optim.SGD(params, lr=ocfg["lr"], momentum=ocfg.get("momentum", 0.9), weight_decay=ocfg.get("weight_decay", 0.0))
    else:
        return optim.AdamW(params, lr=ocfg.get("lr", 1e-3), weight_decay=ocfg.get("weight_decay", 0.01))
