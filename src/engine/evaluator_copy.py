"""
Evaluator for CIFAR10 test set
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from src.datamodules.cifar10 import CIFAR10DataModule
from src.transforms.build import build_model  # 修复：去掉 Project2_Classification
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def evaluate_cifar10(run_dir: str):
    """
    Evaluate model on CIFAR10 official test set
    
    Args:
        run_dir: Path to experiment directory containing cfg_effective.json and best checkpoint
    """
    run_path = Path(run_dir)
    
    # Load config
    cfg_path = run_path / "cfg_effective.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    print(f"Loaded config from {cfg_path}")
    
    # Find best checkpoint
    ckpt_dir = run_path / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("best_*.ckpt"))
    if not ckpt_files:
        ckpt_files = list(ckpt_dir.glob("best_*.pth"))
    if not ckpt_files:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        ckpt_files = list(ckpt_dir.glob("*.pth"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    
    ckpt_path = ckpt_files[0]
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint FIRST to extract class_weights and num_classes
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 🚨 关键修复点 A：从 checkpoint 中提取 class_weights
    # 假设您已经在 trainer.py 中将权重保存到了 checkpoint 的根目录
    class_weights = checkpoint.get("class_weights")
    if class_weights is not None:
        print("Found class_weights in checkpoint.")
        
    # Load model - pass entire cfg to build_model
    num_classes = cfg.get("datamodule", {}).get("num_classes", 10)
    
    # 🚨 关键修复点 B：使用已定义的 class_weights 变量
    model = build_model(cfg, num_classes=num_classes, class_weights=class_weights)
    
    # Load checkpoint state_dict
    if "model" in checkpoint:
        state_dict_to_load = checkpoint["model"]
        print(f"Best accuracy from checkpoint: {checkpoint.get('acc', 'N/A')}")
    elif "model_state" in checkpoint:
        state_dict_to_load = checkpoint["model_state"]
    else:
        state_dict_to_load = checkpoint
    
    # 🚨 关键修复点：使用 strict=False 来忽略 loss_fn.weight 键
    # 这个键在评估时不需要，但存在于旧检查点中
    model.load_state_dict(state_dict_to_load, strict=False) 
    
    model = model.to(device)
    model.eval()
    
    # Setup datamodule for test set
    data_root = cfg.get("task", {}).get("data_root", "./data/cifar10")
    dm_cfg = {
        "data_root": data_root,
        "batch_size": cfg.get("batch_size", 128),
        "num_workers": cfg.get("num_workers", 4),
    }
    datamodule = CIFAR10DataModule(**dm_cfg)
    datamodule.setup(stage="test")
    
    # Build test transforms from config
    from src.transforms.build import build_transforms  # 修复：去掉 Project2_Classification
    aug_cfg = cfg.get("aug", {})
    _, test_transform = build_transforms(aug_cfg)
    
    # Create CIFAR10 test dataset
    test_dataset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=dm_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=dm_cfg.get("num_workers", 4),
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print("\nStarting evaluation...")
    
    # Evaluate
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", file=__import__('sys').stdout):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # Calculate accuracy
    overall_acc = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"{'='*60}")
    
    # Per-class accuracy
    class_names = test_dataset.classes
    print("\nPer-class Accuracy:")
    print(f"{'Class':<15} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 42)
    
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_names[i]:<15} {acc:>9.2f}% {class_correct[i]:>6}/{class_total[i]:<6}")
    
    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR10 model")
    parser.add_argument("--run", type=str, required=True, help="Path to experiment directory")
    args = parser.parse_args()
    
    evaluate_cifar10(args.run)


if __name__ == "__main__":
    main()