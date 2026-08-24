import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from stable_baselines3 import PPO, SAC, A2C

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.domain.twin.models import MachineModel, SensorModel, VirtualRepresentation, EnvironmentModel
from app.rl.environment import DigitalTwinEnv

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'multi_dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_environment(name, df_features):
    machine = MachineModel(name=name)
    machine.parameters["wear"] = 0.0
    for col in df_features.columns:
        machine.add_sensor(SensorModel(name=col, unit="unit", min_threshold=df_features[col].min(), max_threshold=df_features[col].max()))
    
    vrep = VirtualRepresentation(machine=machine)
    env_model = EnvironmentModel(name="TestEnv")
    env_model.update_condition("ambient_temperature", 20.0)
    vrep.environment = env_model
    return vrep

def evaluate_agent(environment, agent, agent_name, episodes=3):
    results = []
    
    for ep in range(episodes):
        obs, info = environment.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        while not (done or truncated):
            start_time = time.perf_counter()
            
            if agent is None:
                # Baseline
                action = np.zeros(environment.action_space.shape)
            else:
                action, _ = agent.predict(obs, deterministic=True)
                
            obs, reward, done, truncated, info = environment.step(action)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            total_reward += reward
            step += 1
            if step % 10 == 0:
                results.append({
                    "Episode": ep + 1,
                    "Step": step,
                    "Agent": agent_name,
                    "Health Score": info["health_score"],
                    "Wear": info["wear"],
                    "Reward": reward,
                    "Latency (ms)": latency_ms
                })
    return pd.DataFrame(results)

def generate_elsevier_plots(combined_df, dataset_name):
    # Elsevier / Publication Quality Settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300
    })
    
    # High-contrast colorblind-friendly palette
    palette = {"Baseline": "#D55E00", "PPO": "#0072B2", "SAC": "#009E73", "A2C": "#CC79A7"}
    markers = {"Baseline": "X", "PPO": "o", "SAC": "s", "A2C": "^"}
    
    # 1. Health Score Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt_sns.lineplot(data=combined_df, x="Step", y="Health Score", hue="Agent", style="Agent", 
                     markers=markers, dashes=False, palette=palette, ax=ax, markevery=5, linewidth=2)
    ax.set_title(f"Health Score Degradation: {dataset_name}", pad=15)
    ax.set_ylabel("Machine Health Score")
    ax.set_xlabel("Simulation Steps")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_health.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Wear Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt_sns.lineplot(data=combined_df, x="Step", y="Wear", hue="Agent", style="Agent", 
                     markers=markers, dashes=False, palette=palette, ax=ax, markevery=5, linewidth=2)
    ax.set_title(f"Accumulated Wear: {dataset_name}", pad=15)
    ax.set_ylabel("Wear Factor")
    ax.set_xlabel("Simulation Steps")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_wear.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(dataset_name, df_features):
    print(f"--- Running Multi-Algorithm Experiment for {dataset_name} ---")
    vrep = create_environment(dataset_name, df_features)
    env = DigitalTwinEnv(vrep, max_steps=200)
    eval_env = DigitalTwinEnv(vrep, max_steps=200)
    
    # 1. PPO
    print("Training PPO...")
    model_ppo = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model_ppo.learn(total_timesteps=3000)
    
    # 2. SAC
    print("Training SAC...")
    model_sac = SAC("MlpPolicy", env, verbose=0, learning_rate=0.001, buffer_size=50000)
    model_sac.learn(total_timesteps=3000)
    
    # 3. A2C
    print("Training A2C...")
    model_a2c = A2C("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model_a2c.learn(total_timesteps=3000)
    
    print("Evaluating Algorithms...")
    baseline_df = evaluate_agent(eval_env, None, "Baseline")
    ppo_df = evaluate_agent(eval_env, model_ppo, "PPO")
    sac_df = evaluate_agent(eval_env, model_sac, "SAC")
    a2c_df = evaluate_agent(eval_env, model_a2c, "A2C")
    
    combined = pd.concat([baseline_df, ppo_df, sac_df, a2c_df])
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.csv")
    combined.to_csv(csv_path, index=False)
    
    # Generate Publication Graphs
    generate_elsevier_plots(combined, dataset_name)
    print(f"Done with {dataset_name}.")

if __name__ == "__main__":
    # 1. AI4I
    ai4i_path = os.path.join(DATA_DIR, 'AI4I_2020', 'ai4i2020.csv')
    if os.path.exists(ai4i_path):
        df = pd.read_csv(ai4i_path)
        features = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
        run_experiment('AI4I_2020', features)
        
    # 2. Gas Turbine
    gt_path = os.path.join(DATA_DIR, 'Gas_Turbine', 'gt_2011.csv')
    if os.path.exists(gt_path):
        df = pd.read_csv(gt_path)
        features = df[['AT', 'AP', 'AH', 'TIT', 'CO', 'NOX']]
        run_experiment('Gas_Turbine', features)
        
    # 3. Hydraulic Systems
    hyd_ps_path = os.path.join(DATA_DIR, 'Hydraulic_Systems', 'PS1.txt')
    hyd_ts_path = os.path.join(DATA_DIR, 'Hydraulic_Systems', 'TS1.txt')
    if os.path.exists(hyd_ps_path) and os.path.exists(hyd_ts_path):
        ps1 = pd.read_csv(hyd_ps_path, sep=r'\s+', header=None).mean(axis=1)
        ts1 = pd.read_csv(hyd_ts_path, sep=r'\s+', header=None).mean(axis=1)
        df_hyd = pd.DataFrame({'Mean_Pressure_PS1': ps1, 'Mean_Temp_TS1': ts1})
        run_experiment('Hydraulic_Systems', df_hyd)
        
    print(f"All real-data experiments completed. Outputs saved to {OUTPUT_DIR}")
