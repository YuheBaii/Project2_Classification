#!/usr/bin/env python3
import os
import random
import json
from pathlib import Path
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader # 确保 DataLoader 全局可用

# Optional: try to import seaborn for nicer heatmap; fallback to sklearn plotting
try:
    import seaborn as sns # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def load_model_and_data(cfg_path, checkpoint_path, device="cuda"):
    """
    加载模型与 dataloader。返回 model, dataloader, ckpt_dict, class_names
    需要项目中提供 build_transforms, build_data, build_model 接口
    """
    # --- 读取 YAML 配置 ---
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        # 假设配置可能是 JSON 或 YAML
        if cfg_path.suffix in ['.yaml', '.yml']:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)

    # --- 导入项目内部构建函数（根据实际路径调整） ---
    try:
        # 确保导入路径正确，这里保持用户原始的导入方式
        from src.transforms.build import build_transforms, build_data, build_model
    except Exception as e:
        raise ImportError(
            "无法导入项目内部构建函数 (build_transforms/build_data/build_model)。\n"
            "请确认模块路径是否正确，或在此脚本中调整导入路径。\n"
            f"原始错误: {e}"
        )

    # 🚨 关键修复点 A：在模型构建和权重检查之前加载 Checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # --- 构建 transforms / datamodule ---
    train_tf, val_tf = build_transforms(cfg.get("augmentation", None))
    # 🚨 关键修复点 B：接收 build_data 返回的 3 个值
    datamodule, class_names, class_weights = build_data(cfg, train_tf, val_tf)

    # --- 获取最终的 class_weights ---
    # 检查 DataModule 是否提供了权重，否则尝试从 Checkpoint 中获取
    # 这解决了 'class_weights is None and 'class_weights' in ckpt' 处的 UnboundLocalError
    if class_weights is None and 'class_weights' in ckpt:
        class_weights = ckpt['class_weights']
        print("Found class_weights in checkpoint and using it.")
    
    # --- 构建模型 ---
    num_classes = len(class_names)
    # 🚨 关键修复点 C：将 class_weights 传递给 build_model
    model = build_model(cfg, num_classes=num_classes, class_weights=class_weights)
    model.to(device)


    # 【关键修改区域：手动构建 dataloader 以确保 transforms 已应用】
    dataloader = None
    target_dataset = None
    
    # 1. 尝试获取 test_dataset (最优先用于最终评估)
    if hasattr(datamodule, "test_dataset") and datamodule.test_dataset is not None:
        target_dataset = datamodule.test_dataset
        print("Using datamodule's test_dataset for inference.")
    # 2. 其次尝试获取 val_dataset
    elif hasattr(datamodule, "val_dataset") and datamodule.val_dataset is not None:
        target_dataset = datamodule.val_dataset
        print("Using datamodule's val_dataset for inference.")
        
    # 如果找到了数据集，则手动创建 DataLoader
    if target_dataset is not None:
        # 从全局导入 DataLoader (确保 NameError 不再发生)
        bs = cfg.get("batch_size", 128)
        nw = cfg.get("num_workers", 4)
        
        dataloader = DataLoader(
            target_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        print(f"Manually created DataLoader with batch_size={bs}, num_workers={nw}")

    # 【回退逻辑：如果 DataModule 没有公开 dataset 属性】
    if dataloader is None:
        print("Dataset properties not found, falling back to dataloader() method.")
        for fn in ["test_dataloader", "val_dataloader", "train_dataloader"]:
            if hasattr(datamodule, fn):
                dl = getattr(datamodule, fn)()
                if dl is not None:
                    dataloader = dl
                    print(f"Fallback: Using dataloader via: {fn}()")
                    break
    
    # 【结束关键修改区域】

    if dataloader is None:
        raise ValueError(
            "无法从 datamodule 获取 dataloader，请检查 dataset 路径是否正确。"
        )
    
    # --- 加载 checkpoint 并处理可能的 key 前缀 ---
    # 确保 class_weights 已在模型构建时被使用
    
    # 从 checkpoint 中提取 class_names（如果存在）
    ckpt_class_names = ckpt.get("class_names", None)
    if ckpt_class_names:
        class_names = ckpt_class_names

    state_dict = ckpt.get("model", ckpt)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 去掉 'module.' 前缀（常见于 DataParallel 保存）
    new_state = {k.replace("module.", "") if k.startswith("module.") else k: v
                 for k, v in state_dict.items()}

    try:
        # 尝试严格加载
        model.load_state_dict(new_state)
    except RuntimeError as e:
        # 🚨 保持 strict=False 修复以应对 loss_fn.weight 等不匹配键
        model.load_state_dict(new_state, strict=False)
        print("Warning: loaded state_dict with strict=False due to mismatch:", e)

    print(f"Loaded model from {checkpoint_path}")
    print(f"Classes: {class_names}")
    print(f"Checkpoint meta: acc={ckpt.get('acc', 'N/A')}, epoch={ckpt.get('epoch', 'N/A')}")

    return model, dataloader, ckpt, class_names

def _batch_to_images_and_labels(batch):
    """兼容不同 dataloader 返回结构：(images, labels) 或 dict"""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    elif isinstance(batch, dict):
        images = batch.get("images") or batch.get("img") or batch.get("input")
        labels = batch.get("labels") or batch.get("targets")
        if images is None or labels is None:
            raise ValueError("无法从 dict batch 中解析 images/labels 字段。")
    else:
        raise ValueError("Unknown batch format from dataloader.")
    return images, labels


def analyze_predictions(model, dataloader, class_names, device='cuda'):
    """分析模型预测结果（自动跳过坏图片或异常 batch）"""
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []

    total = len(dataloader)
    print(f"开始推理，共 {total} 个 batch...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 兼容不同格式
                images, labels = _batch_to_images_and_labels(batch)

                if images is None or labels is None:
                    print(f"⚠️ 第 {batch_idx} 个 batch 无效，跳过。")
                    continue

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # 对于多分类，通常使用 softmax
                if outputs.dim() == 2 and outputs.shape[1] > 1:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                elif outputs.dim() == 2 and outputs.shape[1] == 1: # 可能是二分类，输出为 (N, 1)
                    # 假设 BCEWithLogitsLoss 的输出，使用 sigmoid
                    probs = torch.sigmoid(outputs).squeeze(1) # (N,)
                    preds = (probs > 0.5).long()
                else:
                    # 默认多分类
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)


                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 对于二分类 (N,) 的 probs，需要扩展维度才能用 np.array(all_probs)
                if probs.dim() == 1:
                    # 转换为 (N, 1) 格式，或只存储 (N, 1) 的预测概率
                    all_probs.extend(probs.unsqueeze(1).cpu().numpy())
                else:
                    all_probs.extend(probs.cpu().numpy())

                all_images.extend(images.cpu())

            except Exception as e:
                print(f"⚠️ 跳过第 {batch_idx} 个 batch（可能包含损坏图片或维度错误）：{e}")
                continue

    print(f"推理完成，共成功处理 {len(all_preds)} 张样本")
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制并保存混淆矩阵（支持 seaborn 或 sklearn）"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def visualize_cases(images, labels, preds, probs, class_names,
                    correct=True, num_samples=10, save_path=None, seed=42):
    """可视化正确或错误的样例（保存图片）。images 为 CPU tensor list，labels/preds/probs 为 np.array"""
    random.seed(seed)
    indices = np.where(labels == preds)[0] if correct else np.where(labels != preds)[0]
    if len(indices) == 0:
        print("No samples found for visualization (correct=%s)." % correct)
        return

    num_samples = min(num_samples, len(indices))
    selected = random.sample(indices.tolist(), num_samples)

    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 常用 ImageNet 归一化参数（如你使用不同参数请修改）
    # 注意：您的配置文件使用的正是 ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis("off")
            continue
        idx = selected[i]
        img = images[idx] # tensor C,H,W on CPU
        if isinstance(img, torch.Tensor):
            # 反归一化（假设图像已经被 normalized）
            try:
                img_disp = img * std + mean
            except Exception:
                img_disp = img
            img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()
        else:
            # 如果 image 是 PIL 或 numpy
            img_disp = np.array(img)

        ax.imshow(img_disp)
        true_label = class_names[int(labels[idx])]
        pred_label = class_names[int(preds[idx])]
        
        # 处理 probs 的维度，确保能正确取到置信度
        if probs.ndim == 2:
            conf = float(probs[idx][int(preds[idx])]) * 100.0
        elif probs.ndim == 1: # 二分类 (N,) 只有一列概率
            conf = float(probs[idx]) * 100.0
        else:
            conf = 0.0
        
        color = "green" if correct else "red"
        ax.set_title(f"T:{true_label}\nP:{pred_label}\nConf:{conf:.1f}%", color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{'Correct' if correct else 'Error'} cases saved to {save_path}")
    plt.close()


def analyze_error_statistics(labels, preds, probs, class_names, output_dir=None):
    """打印并保存分类报告与低置信度统计"""
    labels = np.array(labels, dtype=int)
    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)

    # 处理二分类 probs 只有一维的情况
    if probs.ndim == 1:
        # max_probs 保持不变
        max_probs = probs
    else:
        # 多分类 max_probs 保持不变
        max_probs = np.max(probs, axis=1)

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\n" + "="*60)
    print("Classification Report:")
    print(report)
    if output_dir:
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        print(f"Saved classification report to {os.path.join(output_dir, 'classification_report.txt')}")

    # per-class summary
    print("\nPer-class summary:")
    for i, cname in enumerate(class_names):
        mask = labels == i
        total = np.sum(mask)
        correct = np.sum((labels == preds) & mask)
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"  {cname}: total={total}, correct={correct}, acc={acc:.2f}%")
        # 列出被预测为其他类别的计数
        if total > 0:
            for j, other in enumerate(class_names):
                if i == j: continue
                cnt = np.sum(preds[mask] == j)
                if cnt > 0:
                    print(f"    mis -> {other}: {cnt}")

    # 低置信度统计（阈值 0.8）
    low_conf_mask = max_probs < 0.8
    low_count = int(np.sum(low_conf_mask))
    total_count = len(labels)
    acc_low = 100.0 * np.mean(labels[low_conf_mask] == preds[low_conf_mask]) if low_count > 0 else 0.0
    print("\n" + "="*60)
    print(f"Low confidence predictions (<80%): {low_count} / {total_count}")
    print(f"Accuracy on low-confidence set: {acc_low:.2f}%")
    if output_dir:
        with open(os.path.join(output_dir, "low_confidence_stats.txt"), "w") as f:
            f.write(f"low_count={low_count}\n")
            f.write(f"total_count={total_count}\n")
            f.write(f"accuracy_on_low_confidence={acc_low:.4f}\n")
        print(f"Saved low-confidence stats to {os.path.join(output_dir, 'low_confidence_stats.txt')}")

    return {
        "low_count": low_count,
        "total_count": total_count,
        "acc_low": acc_low
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze model errors (confusion matrix, visualize cases, low-conf stats)")
    parser.add_argument("--cfg", required=True, help="Path to cfg_effective.json (config)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output", default="error_analysis", help="Output folder")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of examples to visualize for correct / error")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model, dataloader, ckpt, class_names = load_model_and_data(args.cfg, args.checkpoint, device=args.device)
    preds, labels, probs, images = analyze_predictions(model, dataloader, class_names, device=args.device)

    # 1) 混淆矩阵
    plot_confusion_matrix(labels, preds, class_names, save_path=os.path.join(args.output, "confusion_matrix.png"))

    # 2) 可视化错误与正确样例
    visualize_cases(images, labels, preds, probs, class_names, correct=False, num_samples=args.num_samples,
                    save_path=os.path.join(args.output, "error_cases.png"))
    visualize_cases(images, labels, preds, probs, class_names, correct=True, num_samples=args.num_samples,
                    save_path=os.path.join(args.output, "correct_cases.png"))

    # 3) 低置信度统计与 classification report
    stats = analyze_error_statistics(labels, preds, probs, class_names, output_dir=args.output)

    print("Error analysis finished. Outputs in:", args.output)