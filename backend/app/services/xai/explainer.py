import numpy as np
import shap
from typing import Dict, Any, List

from app.rl.manager import RLManager, MODEL_DIR
from app.rl.environment import DigitalTwinEnv
from app.domain.twin.models import VirtualRepresentation
import os

class XAIExplainer:
    """Provides Explainable AI metrics for the Digital Twin using SHAP and Counterfactuals."""

    @staticmethod
    def get_feature_importance(virtual_rep: VirtualRepresentation, algo: str = "PPO", model_name: str = "twin_agent") -> Dict[str, Any]:
        """Calculates SHAP feature importance for the current state."""
        model_path = os.path.join(MODEL_DIR, f"{model_name}_{algo}.zip")
        if not os.path.exists(model_path):
            return {"error": "RL Model not found for XAI explanation."}
            
        model = RLManager.load_model(algo, model_path)
        env = DigitalTwinEnv(virtual_rep)
        
        # Get baseline observation
        obs = env._get_observation()
        
        # We need a prediction function wrapper for SHAP
        def predict_fn(observations):
            actions = []
            for ob in observations:
                action, _ = model.predict(ob, deterministic=True)
                actions.append(action)
            return np.array(actions)

        # Generating background data (mocked baseline centered at 0 for normalized inputs)
        background = np.zeros((10, obs.shape[0]))
        
        # SHAP KernelExplainer is model agnostic
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Calculate SHAP values for the current single observation
        shap_values = explainer.shap_values(obs.reshape(1, -1))
        
        # Feature names map to environment observation space
        feature_names = env.sensor_keys + ["wear_factor"]
        
        # Format for Plotly UI consumption
        importance = {}
        # shap_values is a list of arrays (one for each action dimension)
        # We'll average the absolute importance across all action outputs for a global feature importance score
        avg_shap = np.mean(np.abs(np.array(shap_values)), axis=0)[0]
        
        for i, name in enumerate(feature_names):
            importance[name] = float(avg_shap[i])
            
        return {
            "feature_importance": importance,
            "base_value": explainer.expected_value.tolist() if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        }

    @staticmethod
    def counterfactual_analysis(virtual_rep: VirtualRepresentation, tweak_sensor: str, tweak_value: float, algo: str = "PPO", model_name: str = "twin_agent") -> Dict[str, Any]:
        """
        What-If analysis: If a sensor value was different, what would the RL agent have done?
        """
        # Baseline action
        original_action = RLManager.get_action(virtual_rep, algo, os.path.join(MODEL_DIR, f"{model_name}_{algo}.zip"))
        
        # Clone state and apply counterfactual
        cf_rep = virtual_rep.clone_state()
        if tweak_sensor in cf_rep.machine.sensors:
            cf_rep.machine.sensors[tweak_sensor].current_value = tweak_value
        elif tweak_sensor == "wear_factor":
            cf_rep.machine.parameters["wear"] = tweak_value
            
        # New action
        cf_action = RLManager.get_action(cf_rep, algo, os.path.join(MODEL_DIR, f"{model_name}_{algo}.zip"))
        
        return {
            "original_action": original_action,
            "counterfactual_action": cf_action,
            "delta": (np.array(cf_action) - np.array(original_action)).tolist()
        }
