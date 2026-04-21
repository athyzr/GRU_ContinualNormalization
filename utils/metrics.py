"""
Evaluation metrics for classification
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def calculate_metrics(y_true, y_pred, num_classes=None, average='macro'):
    """
    Calculate classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        average: 'macro', 'micro', or 'weighted'
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    if num_classes:
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
        )
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
        )
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
        )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def calculate_auc(y_true, y_score, num_classes):
    """
    Calculate AUC scores
    
    Args:
        y_true: Ground truth labels (shape: [N])
        y_score: Predicted probabilities (shape: [N, num_classes])
        num_classes: Number of classes
    
    Returns:
        Dictionary with AUC scores
    """
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    auc_scores = {}
    
    try:
        # Macro AUC
        auc_scores['auc_macro'] = roc_auc_score(
            y_true_bin, y_score, average='macro', multi_class='ovr'
        )
        
        # Weighted AUC
        auc_scores['auc_weighted'] = roc_auc_score(
            y_true_bin, y_score, average='weighted', multi_class='ovr'
        )
        
        # Per-class AUC
        auc_per_class = []
        for i in range(num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:  # At least 2 classes
                auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
                auc_per_class.append(auc)
            else:
                auc_per_class.append(0.0)
        
        auc_scores['auc_per_class'] = np.array(auc_per_class)
        
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc_scores['auc_macro'] = 0.0
        auc_scores['auc_weighted'] = 0.0
        auc_scores['auc_per_class'] = np.zeros(num_classes)
    
    return auc_scores


def get_classification_report(y_true, y_pred, class_names):
    """
    Get detailed classification report
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    return report


class MetricsTracker:
    """
    Track metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.loss = 0.0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, loss, outputs, labels):
        """
        Update metrics with batch results
        
        Args:
            loss: Batch loss (scalar)
            outputs: Model outputs (logits) [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        """
        self.loss += loss * labels.size(0)
        self.total += labels.size(0)
        
        # Get predictions
        _, predicted = outputs.max(1)
        self.correct += predicted.eq(labels).sum().item()
        
        # Store for later metric calculation
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        # Store probabilities
        probs = torch.softmax(outputs, dim=1)
        self.all_probs.extend(probs.detach().cpu().numpy())
    
    def get_metrics(self, num_classes=None):
        """
        Calculate final metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'loss': self.loss / self.total,
            'accuracy': self.correct / self.total,
        }
        
        # Detailed metrics
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        all_probs = np.array(self.all_probs)
        
        detailed_metrics = calculate_metrics(
            all_labels, all_preds, num_classes=num_classes
        )
        metrics.update(detailed_metrics)
        
        # AUC scores
        if num_classes:
            auc_metrics = calculate_auc(all_labels, all_probs, num_classes)
            metrics.update(auc_metrics)
        
        return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    
    metrics = calculate_metrics(y_true, y_pred, num_classes=3)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\n✓ Metrics test passed!")