"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_learning_curves(train_history, val_history, save_path=None):
    """
    Plot training and validation learning curves
    
    Args:
        train_history: Dict with 'loss' and 'acc' lists
        val_history: Dict with 'loss' and 'acc' lists
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, train_history['loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_history['loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_history['acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_history['acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        normalize: If True, normalize confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(metrics, class_names, save_path=None):
    """
    Plot per-class precision, recall, F1
    
    Args:
        metrics: Dict containing per-class metrics
        class_names: List of class names
        save_path: Path to save figure
    """
    precision = metrics.get('precision_per_class', [])
    recall = metrics.get('recall_per_class', [])
    f1 = metrics.get('f1_per_class', [])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='orange')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='green')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics saved to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_score, class_names, save_path=None):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        class_names: List of class names
        save_path: Path to save figure
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, color in enumerate(colors):
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr, 
            color=color, 
            linewidth=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing visualization functions...")
    
    # Test learning curves
    train_history = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'acc': [0.7, 0.75, 0.8, 0.82, 0.85]
    }
    val_history = {
        'loss': [0.6, 0.5, 0.45, 0.4, 0.38],
        'acc': [0.65, 0.7, 0.72, 0.75, 0.76]
    }
    
    plot_learning_curves(train_history, val_history)
    
    print("\n✓ Visualization test passed!")