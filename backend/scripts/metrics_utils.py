import numpy as np
import time
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_classification_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray = None, task_type: str = 'binary') -> Dict[str, float]:
    """Calculates real metrics using sklearn."""
    
    if len(np.unique(true_labels)) < 2:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "Macro-F1": 0.0, "ROC-AUC": 0.5}
        
    accuracy = accuracy_score(true_labels, pred_labels)
    
    if task_type == 'binary':
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        pr_auc = 0.0
        roc_auc = 0.5
        if pred_probs is not None:
            pr_auc = average_precision_score(true_labels, pred_probs)
            roc_auc = roc_auc_score(true_labels, pred_probs)
            
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro-F1": macro_f1,
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc
        }
    else:
        # Multiclass
        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        roc_auc = 0.5
        if pred_probs is not None:
            try:
                # one-vs-rest macro AUC
                roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
            except:
                pass
                
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro-F1": macro_f1,
            "ROC-AUC": roc_auc
        }

if __name__ == "__main__":
    print("Testing metrics...")
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3])
    print(calculate_classification_metrics(y_true, y_pred, y_prob, 'binary'))
