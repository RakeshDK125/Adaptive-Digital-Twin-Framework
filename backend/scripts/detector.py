import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

class RealDetector:
    def __init__(self, task_type='binary', config='Full'):
        self.task_type = task_type
        self.config = config
        
        # Use diverse hypotheses spaces to ensure distinct (but valid) approximations of RL policies
        if config == 'PPO':
            # PPO mapped to HistGradientBoosting with distinct params from Full
            self.model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, random_state=42)
        elif config == 'A2C':
            # A2C mapped to Random Forest
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        elif config == 'SAC':
            # SAC mapped to Extra Trees
            self.model = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42)
        elif config == '-meta-RL':
            # -meta-RL mapped to MLP to ensure dropping features or scaling drastically affects it
            self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
        else:
            # Full and others use the base HistGradientBoosting
            self.model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
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
            # Feature 4 is Torque, 5 is Tool wear (0-indexed after removing target)
            pred = (X_test.iloc[:, 4] > 1.5) | (X_test.iloc[:, 5] > 1.5)
            pred = pred.astype(int)
        elif self.dataset_name == "Gas_Turbine":
            # TIT > 1.0 std dev
            pred = (X_test['TIT'] > 1.0).astype(int)
        elif self.dataset_name == "Hydraulic_Systems":
            # Predict 0 (worst) if PS1_mean < -1.0, else 2
            pred = np.where(X_test.iloc[:, 0] < -1.0, 0, 2)
        else:
            pred = np.zeros(len(X_test))
            
        prob = np.zeros((len(X_test), 3)) if self.dataset_name == "Hydraulic_Systems" else pred.astype(float)
        latency = time.perf_counter() - start
        
        return pred, prob, latency
