#!/usr/bin/env python3
"""
Plot comparative learning curves for CIFAR-10 model performance before and after modifications.
Designed for academic reporting and publication-quality visualizations.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse

# Set professional plotting style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use widely available font
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

def load_history(exp_dir):
    """Load training history data from experiment directory"""
    history_path = Path(exp_dir) / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"✅ Loaded history data from: {exp_dir}")
    print(f"   Training epochs: {len(history.get('train_loss', []))}")
    print(f"   Available metrics: {list(history.keys())}")
    
    return history

def plot_comparison_curves(history_before, history_after, output_path=None):
    """
    Plot comparative learning curves (2x2 layout) for model performance
    before and after modifications
    """
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CIFAR-10 ResNet Model Performance: Before vs After Architecture Modifications\n(Before: Standard ResNet18 vs After: Adapted ResNet18Small)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Ensure consistent epoch range (take minimum length)
    min_epochs = min(
        len(history_before.get('train_loss', [])),
        len(history_after.get('train_loss', []))
    )
    epochs = np.arange(1, min_epochs + 1)
    
    # 1. Training Accuracy Comparison (Top-left)
    ax = axes[0, 0]
    if 'train_accuracy' in history_before and 'train_accuracy' in history_after:
        ax.plot(epochs, history_before['train_accuracy'][:min_epochs], 
                'b-', linewidth=2.5, label='Before Modification (Standard ResNet18)', alpha=0.8)
        ax.plot(epochs, history_after['train_accuracy'][:min_epochs], 
                'r-', linewidth=2.5, label='After Modification (Adapted ResNet18Small)', alpha=0.8)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)  # Accuracy range 0-1
    
    # 2. Validation Accuracy Comparison (Top-right)
    ax = axes[0, 1]
    if 'val_accuracy' in history_before and 'val_accuracy' in history_after:
        ax.plot(epochs, history_before['val_accuracy'][:min_epochs], 
                'b-', linewidth=2.5, label='Before Modification (Standard ResNet18)', alpha=0.8)
        ax.plot(epochs, history_after['val_accuracy'][:min_epochs], 
                'r-', linewidth=2.5, label='After Modification (Adapted ResNet18Small)', alpha=0.8)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 3. Training Loss Comparison (Bottom-left)
    ax = axes[1, 0]
    if 'train_loss' in history_before and 'train_loss' in history_after:
        ax.plot(epochs, history_before['train_loss'][:min_epochs], 
                'b-', linewidth=2.5, label='Before Modification (Standard ResNet18)', alpha=0.8)
        ax.plot(epochs, history_after['train_loss'][:min_epochs], 
                'r-', linewidth=2.5, label='After Modification (Adapted ResNet18Small)', alpha=0.8)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Loss Value')
    ax.set_title('Training Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Use log scale for better visualization of loss changes
    ax.set_yscale('log')
    
    # 4. Validation Loss Comparison (Bottom-right)
    ax = axes[1, 1]
    if 'val_loss' in history_before and 'val_loss' in history_after:
        ax.plot(epochs, history_before['val_loss'][:min_epochs], 
                'b-', linewidth=2.5, label='Before Modification (Standard ResNet18)', alpha=0.8)
        ax.plot(epochs, history_after['val_loss'][:min_epochs], 
                'r-', linewidth=2.5, label='After Modification (Adapted ResNet18Small)', alpha=0.8)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Loss Value')
    ax.set_title('Validation Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for main title
    
    # Save the figure
    if output_path is None:
        output_path = "cifar10_resnet_comparison.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparative plot saved to: {output_path}")
    
    # Display final performance statistics
    print("\n" + "="*60)
    print("Final Performance Statistics (average of last 5 epochs):")
    print("="*60)
    
    if 'val_accuracy' in history_before and 'val_accuracy' in history_after:
        last_5_before = np.mean(history_before['val_accuracy'][-5:])
        last_5_after = np.mean(history_after['val_accuracy'][-5:])
        improvement = last_5_after - last_5_before
        improvement_percent = improvement * 100
        
        print(f"Before modification validation accuracy: {last_5_before:.4f} ({last_5_before*100:.2f}%)")
        print(f"After modification validation accuracy:  {last_5_after:.4f} ({last_5_after*100:.2f}%)")
        print(f"Accuracy improvement: {improvement:.4f} ({improvement_percent:+.2f}%)")
        
        # Additional statistical information
        if 'val_loss' in history_before and 'val_loss' in history_after:
            loss_before = np.mean(history_before['val_loss'][-5:])
            loss_after = np.mean(history_after['val_loss'][-5:])
            loss_reduction = loss_before - loss_after
            print(f"Validation loss reduction: {loss_reduction:.4f} ({loss_reduction/loss_before*100:.1f}% improvement)")
    
    plt.show()
    
    return fig

def plot_individual_curves(history_before, history_after, output_dir="comparison_plots"):
    """Plot detailed learning curves for each model individually"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning curves for the model before modification
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if 'train_accuracy' in history_before:
        plt.plot(history_before['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history_before:
        plt.plot(history_before['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Before: Standard ResNet18 - Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    if 'train_loss' in history_before:
        plt.plot(history_before['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_before:
        plt.plot(history_before['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Before: Standard ResNet18 - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    if 'train_accuracy' in history_after:
        plt.plot(history_after['train_accuracy'], 'g-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history_after:
        plt.plot(history_after['val_accuracy'], 'orange', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('After: Adapted ResNet18Small - Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if 'train_loss' in history_after:
        plt.plot(history_after['train_loss'], 'g-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_after:
        plt.plot(history_after['val_loss'], 'orange', label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('After: Adapted ResNet18Small - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    individual_path = os.path.join(output_dir, "individual_learning_curves.png")
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    print(f"✅ Individual learning curves saved to: {individual_path}")
    plt.close()

def calculate_convergence_metrics(history_before, history_after):
    """Calculate convergence metrics and performance statistics"""
    print("\n" + "="*60)
    print("Convergence Analysis and Performance Metrics")
    print("="*60)
    
    metrics = {}
    
    # Convergence speed analysis (epochs to reach 80% of max accuracy)
    if 'val_accuracy' in history_before:
        max_acc_before = max(history_before['val_accuracy'])
        target_acc_before = 0.8 * max_acc_before
        convergence_epoch_before = next(i for i, acc in enumerate(history_before['val_accuracy']) 
                                      if acc >= target_acc_before)
        metrics['convergence_before'] = convergence_epoch_before
        print(f"Before modification: Reached 80% of max accuracy at epoch {convergence_epoch_before}")
    
    if 'val_accuracy' in history_after:
        max_acc_after = max(history_after['val_accuracy'])
        target_acc_after = 0.8 * max_acc_after
        convergence_epoch_after = next(i for i, acc in enumerate(history_after['val_accuracy']) 
                                     if acc >= target_acc_after)
        metrics['convergence_after'] = convergence_epoch_after
        print(f"After modification:  Reached 80% of max accuracy at epoch {convergence_epoch_after}")
        
        if 'convergence_before' in metrics:
            convergence_improvement = metrics['convergence_before'] - convergence_epoch_after
            print(f"Convergence speed improvement: {convergence_improvement} epochs faster")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Plot comparative learning curves for CIFAR-10 ResNet model performance')
    parser.add_argument('--before', required=True, help='Path to experiment directory before modifications')
    parser.add_argument('--after', required=True, help='Path to experiment directory after modifications')
    parser.add_argument('--output', default='cifar10_resnet_comparison.png', help='Output image path')
    parser.add_argument('--individual', action='store_true', help='Also generate individual learning curves')
    parser.add_argument('--metrics', action='store_true', help='Calculate detailed performance metrics')
    
    args = parser.parse_args()
    
    try:
        # Load historical data
        history_before = load_history(args.before)
        history_after = load_history(args.after)
        
        # Plot comparative curves
        plot_comparison_curves(history_before, history_after, args.output)
        
        # Generate individual learning curves if requested
        if args.individual:
            plot_individual_curves(history_before, history_after)
            
        # Calculate detailed metrics if requested
        if args.metrics:
            calculate_convergence_metrics(history_before, history_after)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please check if the experiment directory paths are correct and history.json files exist")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()