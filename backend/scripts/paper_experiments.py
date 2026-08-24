import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import gymnasium as gym
from stable_baselines3 import PPO

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.domain.twin.models import MachineModel, SensorModel, VirtualRepresentation
from app.rl.environment import DigitalTwinEnv
from app.services.twin.engine import TwinHealthMonitoring

# Paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILE = os.path.join(DATA_DIR, 'ai4i2020.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Download Dataset
if not os.path.exists(DATA_FILE):
    print(f"Downloading dataset from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("Download complete.")

# 2. Load and Analyze Dataset to initialize our Twin
df = pd.read_csv(DATA_FILE)
# Features: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]

machine = MachineModel(name="Milling Machine (AI4I)")
machine.add_sensor(SensorModel(name="air_temp", unit="K", 
                               min_threshold=df['Air temperature [K]'].min(), 
                               max_threshold=df['Air temperature [K]'].max()))
machine.add_sensor(SensorModel(name="process_temp", unit="K", 
                               min_threshold=df['Process temperature [K]'].min(), 
                               max_threshold=df['Process temperature [K]'].max()))
machine.add_sensor(SensorModel(name="rotational_speed", unit="rpm", 
                               min_threshold=df['Rotational speed [rpm]'].min(), 
                               max_threshold=df['Rotational speed [rpm]'].max()))
machine.add_sensor(SensorModel(name="torque", unit="Nm", 
                               min_threshold=df['Torque [Nm]'].min(), 
                               max_threshold=df['Torque [Nm]'].max()))
machine.add_sensor(SensorModel(name="tool_wear", unit="min", 
                               min_threshold=df['Tool wear [min]'].min(), 
                               max_threshold=df['Tool wear [min]'].max()))

virtual_rep = VirtualRepresentation(machine)

# We create two environments: one for training, one for evaluation
env = DigitalTwinEnv(virtual_rep, max_steps=500)
eval_env = DigitalTwinEnv(virtual_rep, max_steps=500)

print("Starting RL Agent Training (PPO)...")
# 3. Train Agent
model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
model.learn(total_timesteps=10000)
print("Training completed.")

# 4. Evaluation Loop
def evaluate_agent(environment, agent, is_baseline=False, episodes=5):
    results = []
    
    for ep in range(episodes):
        obs, info = environment.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        while not (done or truncated):
            if is_baseline:
                # Baseline: Do nothing (actions = 0)
                action = np.zeros(environment.action_space.shape)
            else:
                action, _states = agent.predict(obs, deterministic=True)
                
            obs, reward, done, truncated, info = environment.step(action)
            total_reward += reward
            step += 1
            
            # Record state every 10 steps for time-series plots
            if step % 10 == 0:
                results.append({
                    "Episode": ep + 1,
                    "Step": step,
                    "Agent": "Baseline" if is_baseline else "RL Agent",
                    "Health Score": info["health_score"],
                    "Wear": info["wear"],
                    "Reward": reward
                })
                
    return pd.DataFrame(results)

print("Evaluating Agents...")
rl_results = evaluate_agent(eval_env, model, is_baseline=False)
baseline_results = evaluate_agent(eval_env, None, is_baseline=True)

# 5. Save Tabular Results
combined_results = pd.concat([rl_results, baseline_results])
csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
combined_results.to_csv(csv_path, index=False)
print(f"Tabular results saved to {csv_path}")

# 6. Generate Graphs for the Paper
plt_sns.set_theme(style="darkgrid")

# Graph 1: Health Score over Time
plt.figure(figsize=(10, 6))
plt_sns.lineplot(data=combined_results, x="Step", y="Health Score", hue="Agent", errorbar='sd')
plt.title("Machine Health Score Degradation Over Time")
plt.ylabel("Health Score")
plt.xlabel("Simulation Steps")
plt.ylim(0, 110)
health_plot_path = os.path.join(OUTPUT_DIR, 'health_score_plot.png')
plt.savefig(health_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Graph 2: Cumulative Wear over Time
plt.figure(figsize=(10, 6))
plt_sns.lineplot(data=combined_results, x="Step", y="Wear", hue="Agent", errorbar='sd')
plt.title("Accumulated Wear Over Time")
plt.ylabel("Wear Factor")
plt.xlabel("Simulation Steps")
wear_plot_path = os.path.join(OUTPUT_DIR, 'wear_plot.png')
plt.savefig(wear_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Graphs generated and saved to {OUTPUT_DIR}")
print("Experiment script finished successfully.")
