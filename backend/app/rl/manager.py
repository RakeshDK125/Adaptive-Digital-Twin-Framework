import os
from typing import Dict, Any, Type, Optional
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from app.rl.environment import DigitalTwinEnv
from app.domain.twin.models import VirtualRepresentation

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALGO_MAP: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DQN": DQN
}

class RLManager:
    """Manages training, evaluating, and running inference for RL models."""
    
    @staticmethod
    def train_model(
        virtual_rep: VirtualRepresentation, 
        algo_name: str = "PPO", 
        timesteps: int = 10000,
        model_name: str = "twin_model"
    ) -> str:
        """Trains a new RL model from scratch."""
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Algorithm {algo_name} not supported.")
            
        def env_creator():
            return DigitalTwinEnv(virtual_rep)
            
        # Vectorized environment for faster training
        vec_env = make_vec_env(env_creator, n_envs=4)
        
        model_class = ALGO_MAP[algo_name]
        
        # DQN requires discrete actions, but our env is continuous. 
        # In a full production env we'd have a DiscreteDigitalTwinEnv wrapper for DQN.
        # Assuming PPO/SAC/TD3 for this continuous env.
        
        model = model_class("MlpPolicy", vec_env, verbose=1, tensorboard_log=os.path.join(MODEL_DIR, "logs"))
        
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=os.path.join(MODEL_DIR, "checkpoints"),
            name_prefix=model_name
        )
        
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
        
        save_path = os.path.join(MODEL_DIR, f"{model_name}_{algo_name}.zip")
        model.save(save_path)
        return save_path

    @staticmethod
    def load_model(algo_name: str, model_path: str) -> BaseAlgorithm:
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Algorithm {algo_name} not supported.")
        return ALGO_MAP[algo_name].load(model_path)

    @staticmethod
    def evaluate(virtual_rep: VirtualRepresentation, algo_name: str, model_path: str, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluates a saved policy on the environment."""
        model = RLManager.load_model(algo_name, model_path)
        env = DigitalTwinEnv(virtual_rep)
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
        return {"mean_reward": mean_reward, "std_reward": std_reward}

    @staticmethod
    def get_action(virtual_rep: VirtualRepresentation, algo_name: str, model_path: str) -> list:
        """Inference: Returns the optimal action for the current twin state."""
        model = RLManager.load_model(algo_name, model_path)
        
        # Create a dummy env just to use its observation formatting logic
        # In production, extract `_get_observation` to a standalone function
        env = DigitalTwinEnv(virtual_rep)
        obs = env._get_observation()
        
        action, _states = model.predict(obs, deterministic=True)
        return action.tolist()
        
    @staticmethod
    def online_finetune(virtual_rep: VirtualRepresentation, algo_name: str, model_path: str, timesteps: int = 1000) -> str:
        """Continues learning on an existing model to adapt to drift."""
        model = RLManager.load_model(algo_name, model_path)
        
        def env_creator():
            return DigitalTwinEnv(virtual_rep)
            
        vec_env = make_vec_env(env_creator, n_envs=1)
        model.set_env(vec_env)
        
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(model_path)
        return model_path
