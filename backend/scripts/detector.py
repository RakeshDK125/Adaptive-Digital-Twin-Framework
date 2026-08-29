import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from data_loaders import load_ai4i, load_gas_turbine, load_hydraulic

class RealDetector:
    def __init__(self, task_type='binary'):
        self.task_type = task_type
        # Handle imbalance natively in hist gradient boosting via class_weight if available, 
        # or we just rely on tree splits. Since HistGradientBoosting doesn't support class_weight natively 
        # in all sklearn versions, we'll use standard GradientBoosting if strictly needed, 
        # but Hist is much faster. Let's use MLP or just Hist.
        self.model = HistGradientBoostingClassifier(
            max_iter=100, 
            learning_rate=0.1, 
            random_state=42
        )
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict_and_score(self, X_test):
        start = time.perf_counter()
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test)
        latency = time.perf_counter() - start
        
        # for binary, prob is usually shape (N, 2), we want prob of class 1
        if self.task_type == 'binary' and prob.shape[1] == 2:
            prob = prob[:, 1]
            
        return pred, prob, latency

class RuleBasedBaseline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def fit(self, X_train, y_train):
        pass # No training for rule-based
        
    def predict_and_score(self, X_test):
        start = time.perf_counter()
        
        # Create a dummy probability (0.0 or 1.0) and pred based on physically meaningful rule
        if self.dataset_name == "AI4I_2020":
            # Feature 4 is Torque, 5 is Tool wear (0-indexed after removing target)
            # We'll use feature index safely: 'Tool wear [min]' was index 5 originally, 
            # X_test is scaled, so we just do a rough > 1.5 standard deviations
            pred = (X_test.iloc[:, 4] > 1.5) | (X_test.iloc[:, 5] > 1.5)
            pred = pred.astype(int)
        elif self.dataset_name == "Gas_Turbine":
            # Just predicting based on high TIT (Turbine Inlet Temp) > 1.0 std dev
            pred = (X_test['TIT'] > 1.0).astype(int)
        elif self.dataset_name == "Hydraulic_Systems":
            # Just predict 0 (worst) if PS1_mean is very low
            pred = np.where(X_test.iloc[:, 0] < -1.0, 0, 2)
        else:
            pred = np.zeros(len(X_test))
            
        prob = np.zeros((len(X_test), 3)) if self.dataset_name == "Hydraulic_Systems" else pred.astype(float)
        latency = time.perf_counter() - start
        
        return pred, prob, latency

if __name__ == "__main__":
    data = load_ai4i(42)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    
    det = RealDetector('binary')
    det.fit(X_train, y_train)
    pred, prob, lat = det.predict_and_score(X_test)
    print(f"AI4I Detector latency: {lat:.6f}s")
    
    rb = RuleBasedBaseline("AI4I_2020")
    pred_rb, prob_rb, lat_rb = rb.predict_and_score(X_test)
    print(f"AI4I Rule-based latency: {lat_rb:.6f}s")
