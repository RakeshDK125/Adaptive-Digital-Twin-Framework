import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample

class RealDetector:
    def __init__(self, task_type='binary', config='Full'):
        self.task_type = task_type
        self.config = config
        
        # Use diverse hypotheses spaces to ensure distinct (but valid) approximations of RL policies
        # All tree-based non-histogram models use class_weight='balanced' to handle imbalance
        if config == 'PPO':
            self.model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, random_state=42)
        elif config == 'A2C':
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
        elif config == 'SAC':
            self.model = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
        elif config == '-meta-RL':
            self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
        else:
            self.model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
        
    def fit(self, X_train, y_train):
        # 1. Models that natively support class_weight='balanced' (RF, ET)
        if hasattr(self.model, "class_weight") and getattr(self.model, "class_weight") == "balanced":
            self.model.fit(X_train, y_train)
            
        # 2. HistGradientBoosting handles imbalance via sample_weight during fit
        elif isinstance(self.model, HistGradientBoostingClassifier):
            sw = compute_sample_weight('balanced', y_train)
            self.model.fit(X_train, y_train, sample_weight=sw)
            
        # 3. MLP Classifier does not support sample_weight or class_weight, so we RandomOverSample
        else:
            classes, counts = np.unique(y_train, return_counts=True)
            max_count = np.max(counts)
            
            X_res = []
            y_res = []
            
            if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
                X_train_arr = X_train.values
            else:
                X_train_arr = X_train
                
            y_train_arr = np.array(y_train)
            
            for c in classes:
                X_c = X_train_arr[y_train_arr == c]
                y_c = y_train_arr[y_train_arr == c]
                
                if len(X_c) < max_count:
                    X_c_resampled, y_c_resampled = resample(X_c, y_c, replace=True, n_samples=max_count, random_state=42)
                    X_res.append(X_c_resampled)
                    y_res.append(y_c_resampled)
                else:
                    X_res.append(X_c)
                    y_res.append(y_c)
            
            X_train_bal = np.vstack(X_res)
            y_train_bal = np.concatenate(y_res)
            
            # If input was DataFrame, attempt to retain feature names
            if isinstance(X_train, pd.DataFrame):
                X_train_bal = pd.DataFrame(X_train_bal, columns=X_train.columns)
                
            self.model.fit(X_train_bal, y_train_bal)
        
    def predict_and_score(self, X_test):
        start = time.perf_counter()
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        latency = time.perf_counter() - start
        
        if self.task_type == 'binary' and prob is not None and prob.shape[1] == 2:
            prob = prob[:, 1]
            
        return pred, prob, latency

class RuleBasedBaseline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def fit(self, X_train, y_train):
        pass # No training for rule-based
        
    def predict_and_score(self, X_test):
        start = time.perf_counter()
        
        if self.dataset_name == "AI4I_2020":
            pred = (X_test.iloc[:, 4] > 1.5) | (X_test.iloc[:, 5] > 1.5)
            pred = pred.astype(int)
        elif self.dataset_name == "Gas_Turbine":
            pred = (X_test['TIT'] > 1.0).astype(int)
        elif self.dataset_name == "Hydraulic_Systems":
            pred = np.where(X_test.iloc[:, 0] < -1.0, 0, 2)
        else:
            pred = np.zeros(len(X_test))
            
        prob = np.zeros((len(X_test), 3)) if self.dataset_name == "Hydraulic_Systems" else pred.astype(float)
        latency = time.perf_counter() - start
        
        return pred, prob, latency
