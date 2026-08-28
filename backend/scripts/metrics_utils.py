import numpy as np
import time
from typing import Dict, Any, List

def calculate_classification_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """Calculates Accuracy, Precision, Recall, and Macro-F1."""
    # Assuming binary classification for anomaly/fault detection (1 = fault, 0 = normal)
    # Using small epsilon to avoid division by zero
    eps = 1e-9
    
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    # Class 1 (Anomaly) metrics
    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + eps)
    
    # Class 0 (Normal) metrics
    precision_0 = tn / (tn + fn + eps)
    recall_0 = tn / (tn + fp + eps)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + eps)
    
    # Macro F1
    macro_f1 = (f1_1 + f1_0) / 2.0
    
    # Assuming positive class for general precision/recall reporting
    return {
        "Accuracy": accuracy,
        "Precision": precision_1,
        "Recall": recall_1,
        "Macro-F1": macro_f1
    }

def calculate_auroc(true_labels: np.ndarray, pred_probs: np.ndarray) -> float:
    """Simplified AUROC approximation (if sklearn is unavailable)."""
    try:
        from sklearn.metrics import roc_auc_score
        # Check if only one class exists
        if len(np.unique(true_labels)) < 2:
            return 0.5
        return roc_auc_score(true_labels, pred_probs)
    except Exception:
        # Fallback approximation if sklearn not installed, though it should be.
        # This sorts by probability and calculates trapezoidal area.
        desc_score_indices = np.argsort(pred_probs, kind="mergesort")[::-1]
        y_true = true_labels[desc_score_indices]
        
        tps = np.cumsum(y_true)
        fps = np.cumsum(1 - y_true)
        
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
        
        # prepend 0s
        tpr = np.r_[0, tpr]
        fpr = np.r_[0, fpr]
        
        auroc = np.trapz(tpr, fpr)
        return float(auroc)

def simulate_latency() -> float:
    """Simulates decision latency for a robust RL/Agentic pipeline."""
    # Base inference latency + slight noise
    return np.random.uniform(0.015, 0.050)

def evaluate_predictions(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Given a list of step results (from an evaluation loop), extract metrics.
    Requires 'True_Anomaly' and 'Pred_Anomaly' keys in the results.
    """
    if not results:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "Macro-F1": 0.0}
        
    true_labels = np.array([r.get("True_Anomaly", 0) for r in results])
    pred_labels = np.array([r.get("Pred_Anomaly", 0) for r in results])
    
    metrics = calculate_classification_metrics(true_labels, pred_labels)
    
    # Add average latency if logged
    latencies = [r.get("Latency", 0.0) for r in results if "Latency" in r]
    if latencies:
        metrics["Latency"] = np.mean(latencies)
        
    return metrics
