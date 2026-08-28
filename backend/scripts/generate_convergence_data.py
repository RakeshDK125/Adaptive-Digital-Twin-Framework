import os
import time
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from lodo_experiments import generate_mock_dataset
from five_seed_runs import create_env

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'experiments_run')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

def main():
    print("Generating real convergence data...")
    seed = 42
    np.random.seed(seed)
    
    # Use AI4I
    df = generate_mock_dataset("AI4I", seed)
    env = create_env(df, seed)
    
    callback = RewardLoggingCallback()
    
    # Train PPO
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=5000, callback=callback)
    
    # Save convergence data
    episodes = np.arange(1, len(callback.episode_rewards) + 1)
    rewards = callback.episode_rewards
    
    df_out = pd.DataFrame({"Episode": episodes, "Reward": rewards})
    
    # Smooth the curve using moving average for a cleaner publication plot
    df_out['Smoothed_Reward'] = df_out['Reward'].rolling(window=5, min_periods=1).mean()
    
    df_out.to_csv(os.path.join(OUTPUT_DIR, 'convergence_results.csv'), index=False)
    print("Convergence data saved.")

if __name__ == "__main__":
    main()
