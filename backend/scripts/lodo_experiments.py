import os
import pandas as pd
import numpy as np
from data_loaders import load_ai4i, load_gas_turbine, load_hydraulic
from detector import RealDetector
from metrics_utils import calculate_classification_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

def run_lodo():
    print("Starting LODO Evaluations...")
    # LODO requires a shared domain-agnostic feature representation.
    # Since they have completely different feature dimensions, we pad/truncate them to a common dim (e.g. 6)
    
    datasets = {
        "AI4I": load_ai4i,
        "Gas": load_gas_turbine,
        "Hydraulic": load_hydraulic
    }
    
    # We will simulate the scenarios
    scenarios = [
        ("Scenario_A", ["Gas", "Hydraulic"], "AI4I"),
        ("Scenario_B", ["AI4I", "Hydraulic"], "Gas"),
        ("Scenario_C", ["AI4I", "Gas"], "Hydraulic")
    ]
    
    lodo_results = []
    
    for scenario_name, train_ds_names, test_ds_name in scenarios:
        for regime in ["zero_shot", "few_shot"]:
            for seed in [13, 42, 87, 123, 2024]:
                
                # We mock the shared representation by just training on the target test dataset 
                # (which is mathematically what the LODO transfer effectively evaluates when 
                # domains are completely disjoint and we enforce a domain-agnostic projection).
                # To enforce "NO leakage", we would strictly train a model on padded X_train of A+B, 
                # and evaluate on padded X_test of C. Let's do that!
                
                # Load Test dataset C
                data_C = datasets[test_ds_name](seed)
                X_test_C, y_test_C = data_C["test"]
                task_C = data_C["task_type"]
                
                # Create shared dimensionality (pad to 10)
                def pad_df(df, target_dim=10):
                    arr = df.values
                    if arr.shape[1] < target_dim:
                        pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]))
                        arr = np.hstack([arr, pad])
                    else:
                        arr = arr[:, :target_dim]
                    return arr
                    
                X_test_C_padded = pad_df(X_test_C)
                
                # Load Train datasets A and B
                X_train_A, y_train_A = datasets[train_ds_names[0]](seed)["train"]
                X_train_B, y_train_B = datasets[train_ds_names[1]](seed)["train"]
                
                # Because labels might be multiclass vs binary, we simplify to binary for transfer
                y_train_A = (y_train_A > 0).astype(int).values if hasattr(y_train_A, 'values') else (y_train_A > 0).astype(int)
                y_train_B = (y_train_B > 0).astype(int).values if hasattr(y_train_B, 'values') else (y_train_B > 0).astype(int)
                y_test_C_bin = (y_test_C > 0).astype(int).values if hasattr(y_test_C, 'values') else (y_test_C > 0).astype(int)
                
                X_train_A_padded = pad_df(X_train_A)
                X_train_B_padded = pad_df(X_train_B)
                
                if regime == "zero_shot":
                    # Train on A + B only
                    X_train_shared = np.vstack([X_train_A_padded, X_train_B_padded])
                    y_train_shared = np.concatenate([y_train_A, y_train_B])
                else: # few_shot
                    # Train on A + B + a small portion of C train
                    X_train_C, y_train_C = data_C["train"]
                    X_train_C_padded = pad_df(X_train_C)
                    y_train_C = (y_train_C > 0).astype(int).values if hasattr(y_train_C, 'values') else (y_train_C > 0).astype(int)
                    
                    # 10% few shot
                    num_samples = max(1, int(0.1 * len(X_train_C_padded)))
                    X_train_shared = np.vstack([X_train_A_padded, X_train_B_padded, X_train_C_padded[:num_samples]])
                    y_train_shared = np.concatenate([y_train_A, y_train_B, y_train_C[:num_samples]])
                    
                # Train the real detector
                det = RealDetector('binary')
                det.fit(X_train_shared, y_train_shared)
                
                pred, prob, lat = det.predict_and_score(X_test_C_padded)
                
                metrics = calculate_classification_metrics(y_test_C_bin, pred, prob, 'binary')
                
                lodo_results.append({
                    "Scenario": scenario_name,
                    "Regime": regime,
                    "Seed": seed,
                    "Macro-F1": metrics["Macro-F1"],
                    "ROC-AUC": metrics["ROC-AUC"]
                })
                
    df = pd.DataFrame(lodo_results)
    out_path = os.path.join(RESULTS_DIR, "lodo_real.csv")
    df.to_csv(out_path, index=False)
    print(f"LODO real evaluation saved to {out_path}")

if __name__ == "__main__":
    run_lodo()
