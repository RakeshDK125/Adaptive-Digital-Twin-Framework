import numpy as np
import time
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_classification_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray = None, task_type: str = 'binary') -> Dict[str, Any]:
    """Calculates real metrics using sklearn, including granular minority and baseline metrics."""
    
    # Calculate prevalence (for binary it's the 1 class, for multiclass it's a dict or min class)
    classes, counts = np.unique(true_labels, return_counts=True)
    if len(classes) == 0:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "Macro-F1": 0.0, "ROC-AUC": 0.5, "PR-AUC": 0.0, "Minority-Recall": 0.0, "Majority-Baseline-F1": 0.0, "Prevalence": 0.0}
        
    majority_class = classes[np.argmax(counts)]
    prevalence = np.min(counts) / len(true_labels) if len(classes) > 1 else 0.0
    
    # Compute Majority Baseline F1 (predicting majority class for everything)
    majority_preds = np.full_like(true_labels, majority_class)
    majority_f1 = f1_score(true_labels, majority_preds, average='macro', zero_division=0)
    
    if len(classes) < 2:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "Macro-F1": 0.0, "ROC-AUC": 0.5, "PR-AUC": 0.0, "Minority-Recall": 0.0, "Majority-Baseline-F1": majority_f1, "Prevalence": prevalence}
        
    accuracy = accuracy_score(true_labels, pred_labels)
    
    if task_type == 'binary':
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        # Per class metrics
        recalls = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        min_recall = np.min(recalls)
        
        pr_auc = 0.0
        roc_auc = 0.5
        if pred_probs is not None:
            try:
                pr_auc = average_precision_score(true_labels, pred_probs)
                roc_auc = roc_auc_score(true_labels, pred_probs)
            except:
                pass
            
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro-F1": macro_f1,
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc,
            "Minority-Recall": min_recall,
            "Majority-Baseline-F1": majority_f1,
            "Prevalence": prevalence
        }
    else:
        # Multiclass
        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        # Per class metrics
        recalls = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        min_recall = np.min(recalls)
        
        roc_auc = 0.5
        pr_auc = 0.0
        
        if pred_probs is not None:
            try:
                roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
                
                # Approximate PR-AUC for multiclass (macro average over one-vs-rest)
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(true_labels, classes=classes)
                if y_bin.shape[1] == 1:
                    y_bin = np.hstack([1 - y_bin, y_bin])
                
                # Check shape matching
                if pred_probs.shape == y_bin.shape:
                    pr_auc = average_precision_score(y_bin, pred_probs, average='macro')
            except Exception as e:
                pass
                
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro-F1": macro_f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Minority-Recall": min_recall,
            "Majority-Baseline-F1": majority_f1,
            "Prevalence": prevalence
        }

if __name__ == "__main__":
    print("Testing metrics...")
    y_true = np.array([0, 1, 0, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.1, 0.1, 0.1])
    print(calculate_classification_metrics(y_true, y_pred, y_prob, 'binary'))
