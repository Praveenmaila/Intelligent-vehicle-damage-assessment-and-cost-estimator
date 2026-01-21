"""
Evaluation Metrics Module
Implements metrics from research paper
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray, 
                num_classes: int, ignore_background: bool = True) -> Dict:
    """
    Compute Intersection over Union (IoU) for semantic segmentation.
    
    Args:
        pred_mask: Predicted segmentation (H x W)
        true_mask: Ground truth segmentation (H x W)
        num_classes: Number of classes
        ignore_background: Whether to exclude background (class 0) from mIoU
        
    Returns:
        Dictionary with per-class IoU and mean IoU
    """
    ious = []
    class_ious = {}
    
    start_class = 1 if ignore_background else 0
    
    for class_id in range(start_class, num_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        
        if union == 0:
            # Class not present in ground truth or prediction
            iou = np.nan
        else:
            iou = intersection / union
            ious.append(iou)
        
        class_ious[class_id] = float(iou) if not np.isnan(iou) else None
    
    # Mean IoU (ignoring NaN values)
    miou = np.nanmean(ious) if ious else 0.0
    
    return {
        'mean_iou': float(miou),
        'class_ious': class_ious,
        'valid_classes': len([x for x in ious if not np.isnan(x)])
    }


def compute_dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray,
                             num_classes: int) -> Dict:
    """
    Compute Dice coefficient for segmentation.
    
    Args:
        pred_mask: Predicted segmentation
        true_mask: Ground truth segmentation
        num_classes: Number of classes
        
    Returns:
        Dictionary with Dice scores
    """
    dice_scores = []
    class_dice = {}
    
    for class_id in range(1, num_classes):  # Skip background
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = np.logical_and(pred_class, true_class).sum()
        pred_sum = pred_class.sum()
        true_sum = true_class.sum()
        
        if pred_sum + true_sum == 0:
            dice = np.nan
        else:
            dice = (2.0 * intersection) / (pred_sum + true_sum)
            dice_scores.append(dice)
        
        class_dice[class_id] = float(dice) if not np.isnan(dice) else None
    
    mean_dice = np.nanmean(dice_scores) if dice_scores else 0.0
    
    return {
        'mean_dice': float(mean_dice),
        'class_dice': class_dice
    }


def compute_pixel_accuracy(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred_mask: Predicted segmentation
        true_mask: Ground truth segmentation
        
    Returns:
        Pixel accuracy as float
    """
    correct = (pred_mask == true_mask).sum()
    total = pred_mask.size
    return float(correct / total)


def compute_confusion_matrix(pred_mask: np.ndarray, true_mask: np.ndarray,
                             num_classes: int, class_names: List[str] = None,
                             normalize: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Compute confusion matrix for segmentation.
    
    Args:
        pred_mask: Predicted segmentation (H x W)
        true_mask: Ground truth segmentation (H x W)
        num_classes: Number of classes
        class_names: List of class names
        normalize: Whether to normalize by row (true class)
        
    Returns:
        Tuple of (confusion_matrix, metrics_dict)
    """
    # Flatten masks
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    # Compute per-class metrics
    class_metrics = {}
    for i in range(num_classes):
        tp = cm[i, i] if not normalize else 0  # True positives
        fn = cm[i, :].sum() - tp  # False negatives
        fp = cm[:, i].sum() - tp  # False positives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = class_names[i] if class_names else f"Class_{i}"
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    return cm, class_metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         save_path: str = None, figsize=(12, 10)):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def evaluate_segmentation_model(model, data_loader, device, num_classes: int,
                                class_names: List[str] = None) -> Dict:
    """
    Comprehensive evaluation of segmentation model.
    
    Args:
        model: PyTorch segmentation model
        data_loader: DataLoader for evaluation
        device: Device for computation
        num_classes: Number of classes
        class_names: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_ious = []
    all_dice = []
    all_pixel_acc = []
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.cpu().numpy()
            
            # Forward pass
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Compute metrics for each sample in batch
            for pred, mask in zip(preds, masks):
                # IoU
                iou_result = compute_iou(pred, mask, num_classes)
                all_ious.append(iou_result['mean_iou'])
                
                # Dice
                dice_result = compute_dice_coefficient(pred, mask, num_classes)
                all_dice.append(dice_result['mean_dice'])
                
                # Pixel accuracy
                pixel_acc = compute_pixel_accuracy(pred, mask)
                all_pixel_acc.append(pixel_acc)
                
                # Store for confusion matrix
                all_preds.extend(pred.flatten().tolist())
                all_targets.extend(mask.flatten().tolist())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Aggregate results
    results = {
        'mean_iou': float(np.mean(all_ious)),
        'std_iou': float(np.std(all_ious)),
        'mean_dice': float(np.mean(all_dice)),
        'std_dice': float(np.std(all_dice)),
        'pixel_accuracy': float(np.mean(all_pixel_acc)),
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized
    }
    
    return results


def compute_vehicle_detection_metrics(model, data_loader, device) -> Dict:
    """
    Evaluation metrics for binary vehicle detection.
    
    Args:
        model: Vehicle detection model
        data_loader: DataLoader for evaluation
        device: Device for computation
        
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels_np = labels.cpu().numpy()
            
            # Forward pass
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_targets.extend(labels_np.tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    accuracy = (all_preds == all_targets).mean()
    
    # For binary classification
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def bootstrap_confidence_interval(metric_values: List[float], 
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_values: List of metric values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n_samples = len(metric_values)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(metric_values, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return float(lower), float(upper)


if __name__ == '__main__':
    print("Evaluation Metrics Module")
    print("=" * 60)
    
    # Example: test IoU computation
    pred = np.array([[0, 1, 1], [1, 2, 2], [2, 2, 0]])
    target = np.array([[0, 1, 1], [1, 1, 2], [2, 2, 0]])
    
    iou_result = compute_iou(pred, target, num_classes=3)
    print(f"Example mIoU: {iou_result['mean_iou']:.4f}")
    print(f"Class IoUs: {iou_result['class_ious']}")
    
    print("\n✓ Evaluation module ready")
