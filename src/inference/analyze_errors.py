
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_model_and_predict(checkpoint_path, datamodule, device='cuda'):
    """加载模型并进行预测"""
    from src.transforms.build import build_model
    
    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt.get('class_names', ['cat', 'dog'])
    
    # 重建模型（需要传入正确的配置）
    # 这里需要根据实际情况调整
    print(f"Loaded model with classes: {class_names}")
    print(f"Best accuracy: {ckpt.get('acc', 'N/A')}")
    
    return ckpt, class_names

def analyze_predictions(model, dataloader, class_names, device='cuda'):
    """分析模型预测结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def visualize_error_cases(images, labels, preds, probs, class_names, 
                         num_samples=10, save_path=None):
    """可视化错误分类的案例"""
    # 找出错误分类的样本
    error_indices = np.where(labels != preds)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    # 随机选择一些错误样本
    num_samples = min(num_samples, len(error_indices))
    selected_indices = np.random.choice(error_indices, num_samples, replace=False)
    
    # 绘制
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx >= num_samples:
            ax.axis('off')
            continue
            
        img_idx = selected_indices[idx]
        img = images[img_idx]
        
        # 反标准化（如果需要）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # 转换为可显示的格式
        img_np = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        true_label = class_names[labels[img_idx]]
        pred_label = class_names[preds[img_idx]]
        confidence = probs[img_idx][preds[img_idx]] * 100
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                    color='red', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error cases saved to {save_path}")
    plt.close()

def visualize_correct_cases(images, labels, preds, probs, class_names, 
                           num_samples=10, save_path=None):
    """可视化正确分类的案例"""
    # 找出正确分类的样本
    correct_indices = np.where(labels == preds)[0]
    
    if len(correct_indices) == 0:
        print("No correct predictions found!")
        return
    
    # 随机选择一些正确样本
    num_samples = min(num_samples, len(correct_indices))
    selected_indices = np.random.choice(correct_indices, num_samples, replace=False)
    
    # 绘制
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx >= num_samples:
            ax.axis('off')
            continue
            
        img_idx = selected_indices[idx]
        img = images[img_idx]
        
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        img_np = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        label = class_names[labels[img_idx]]
        confidence = probs[img_idx][preds[img_idx]] * 100
        
        ax.set_title(f'{label}\nConf: {confidence:.1f}%',
                    color='green', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correct cases saved to {save_path}")
    plt.close()

def analyze_error_statistics(labels, preds, probs, class_names):
    """分析错误统计信息"""
    from sklearn.metrics import classification_report
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(labels, preds, target_names=class_names))
    
    # 按类别分析错误
    print("\n" + "="*60)
    print("Per-Class Error Analysis:")
    print("="*60)
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_correct = np.sum((labels == preds) & class_mask)
        class_total = np.sum(class_mask)
        class_acc = class_correct / class_total if class_total > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Total samples: {class_total}")
        print(f"  Correct: {class_correct}")
        print(f"  Accuracy: {class_acc*100:.2f}%")
        
        # 找出误分类为其他类别的数量
        misclassified = labels[class_mask & (labels != preds)]
        if len(misclassified) > 0:
            print(f"  Misclassified as:")
            for j, other_class in enumerate(class_names):
                if i != j:
                    count = np.sum(preds[class_mask & (labels != preds)] == j)
                    if count > 0:
                        print(f"    {other_class}: {count}")
    
    # 分析低置信度预测
    print("\n" + "="*60)
    print("Low Confidence Predictions (< 80%):")
    print("="*60)
    max_probs = np.max(probs, axis=1)
    low_conf_mask = max_probs < 0.8
    print(f"Total low confidence: {np.sum(low_conf_mask)} / {len(labels)}")
    print(f"Accuracy on low confidence: {np.mean(labels[low_conf_mask] == preds[low_conf_mask])*100:.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--cfg", action="append", required=True, help="Config files")
    parser.add_argument("--output", type=str, default="error_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # 这里需要实现完整的分析流程
    print("Error analysis tool - Implementation needed")
