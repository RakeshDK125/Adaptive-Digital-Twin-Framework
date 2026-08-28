import os
import time
import random
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from five_seed_runs import load_data, create_env
from lodo_experiments import evaluate_regime
from metrics_utils import calculate_classification_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'experiments_run')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [13, 42, 87, 123, 2024]
BUDGET = 2

def sample_sac_params():
    return {
        "learning_rate": 10 ** np.random.uniform(-4, -2),
        "batch_size": random.choice([64, 128, 256]),
        "gamma": np.random.uniform(0.9, 0.999),
        "tau": np.random.uniform(0.001, 0.02),
        "buffer_size": 50000,
        "ent_coef": random.choice(["auto", 10 ** np.random.uniform(-2, -0.7)])
    }

def sample_a2c_params():
    return {
        "learning_rate": 10 ** np.random.uniform(-4, -2),
        "n_steps": random.choice([5, 10, 20]),
        "gamma": np.random.uniform(0.9, 0.99),
        "gae_lambda": np.random.uniform(0.8, 0.99),
        "ent_coef": np.random.uniform(0.0, 0.01),
        "vf_coef": np.random.uniform(0.25, 0.5)
    }

def sample_ppo_params(): # Simulating the meta-PPO RL params
    return {
        "learning_rate": 10 ** np.random.uniform(-4, -2),
        "n_steps": random.choice([1024, 2048, 4096]),
        "batch_size": random.choice([32, 64, 128]),
        "gamma": np.random.uniform(0.9, 0.999),
        "gae_lambda": np.random.uniform(0.9, 0.99),
        "ent_coef": np.random.uniform(0.0, 0.01),
        "vf_coef": np.random.uniform(0.25, 0.75)
    }

def run_tuning_for_algo(algo_name: str, train_env, val_env, test_env, seed: int):
    best_val_f1 = -1
    best_params = None
    best_agent = None
    
    print(f"    Tuning {algo_name} (Budget: {BUDGET} trials)...")
    
    for trial in range(BUDGET):
        if algo_name == "SAC":
            params = sample_sac_params()
            agent = SAC("MlpPolicy", train_env, verbose=0, **params)
        elif algo_name == "A2C":
            params = sample_a2c_params()
            agent = A2C("MlpPolicy", train_env, verbose=0, **params)
        elif algo_name == "PPO":
            params = sample_ppo_params()
            agent = PPO("MlpPolicy", train_env, verbose=0, **params)
            
        # Fast train for evaluation
        agent.learn(total_timesteps=500)
        
        # Evaluate on validation
        results = evaluate_regime(val_env, agent, regime="zero-shot", seed=seed)
        true_labels = np.array([r["True_Anomaly"] for r in results])
        pred_labels = np.array([r["Pred_Anomaly"] for r in results])
        
        metrics = calculate_classification_metrics(true_labels, pred_labels)
        val_f1 = metrics["Macro-F1"]
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
            best_agent = agent
            
    # Final Evaluation on Test Set with Best Agent
    test_results = evaluate_regime(test_env, best_agent, regime="zero-shot", seed=seed)
    true_labels = np.array([r["True_Anomaly"] for r in test_results])
    pred_labels = np.array([r["Pred_Anomaly"] for r in test_results])
    
    final_metrics = calculate_classification_metrics(true_labels, pred_labels)
    
    return final_metrics, best_params

def main():
    all_results = []
    print("Starting Equal-Budget Baseline Tuning...")
    
    algorithms = ["PPO", "A2C", "SAC"]
    
    for seed in SEEDS:
        print(f"  Seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        
        train_df, val_df, test_df = load_data(seed)
        train_env = create_env(train_df, seed)
        val_env = create_env(val_df, seed)
        test_env = create_env(test_df, seed)
        
        for algo in algorithms:
            metrics, best_params = run_tuning_for_algo(algo, train_env, val_env, test_env, seed)
            
            row = {
                "Algorithm": algo,
                "Seed": seed,
                "Test_Accuracy": metrics["Accuracy"],
                "Test_Precision": metrics["Precision"],
                "Test_Recall": metrics["Recall"],
                "Test_Macro-F1": metrics["Macro-F1"],
                "Best_Hyperparameters": str(best_params)
            }
            all_results.append(row)
            
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "tuning_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Done! Tuning results saved to {csv_path}")

if __name__ == "__main__":
    main()
