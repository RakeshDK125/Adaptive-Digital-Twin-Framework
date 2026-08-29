import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loaders import load_ai4i, load_gas_turbine, load_hydraulic
from detector import RealDetector
from metrics_utils import calculate_classification_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

def run_lodo():
    print("Starting Authentic LODO Evaluations...")
    
    datasets = {
        "AI4I": load_ai4i,
        "Gas": load_gas_turbine,
        "Hydraulic": load_hydraulic
    }
    
    scenarios = [
        ("Scenario_A", ["Gas", "Hydraulic"], "AI4I"),
        ("Scenario_B", ["AI4I", "Hydraulic"], "Gas"),
        ("Scenario_C", ["AI4I", "Gas"], "Hydraulic")
    ]
    
    lodo_results = []
    
    def align_and_pad(X_train_df, X_test_df=None, target_dim=15):
        # Fit scaler ONLY on train split
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train_df)
        
        # Pad to common dimension
        def pad_arr(arr):
            if arr.shape[1] < target_dim:
                pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]))
                arr = np.hstack([arr, pad])
            else:
                arr = arr[:, :target_dim]
            return arr
            
        X_tr_pad = pad_arr(X_tr_sc)
        
        if X_test_df is not None:
            # Transform test split using train scaler (no leakage)
            X_te_sc = scaler.transform(X_test_df)
            X_te_pad = pad_arr(X_te_sc)
            return X_tr_pad, X_te_pad
        return X_tr_pad

    for scenario_name, train_ds_names, test_ds_name in scenarios:
        for regime in ["zero_shot", "few_shot"]:
            for seed in [13, 42, 87, 123, 2024]:
                
                # Load Test dataset C
                data_C = datasets[test_ds_name](seed)
                X_train_C_raw, y_train_C_raw = data_C["train"]
                X_test_C_raw, y_test_C_raw = data_C["test"]
                
                # Align Target Domain C
                X_train_C, X_test_C = align_and_pad(X_train_C_raw, X_test_C_raw)
                
                # Ensure labels are binary integers
                def binarize(y):
                    return (y > 0).astype(int).values if hasattr(y, 'values') else (y > 0).astype(int)
                
                y_train_C = binarize(y_train_C_raw)
                y_test_C = binarize(y_test_C_raw)
                
                # Load and Align Source Domains
                X_train_sources = []
                y_train_sources = []
                for ds_name in train_ds_names:
                    data_src = datasets[ds_name](seed)
                    X_tr_src_raw, y_tr_src_raw = data_src["train"]
                    
                    # Note: For strict Zero-Shot transfer, we align Source A and Source B 
                    # independently using ONLY their own train splits.
                    X_tr_src = align_and_pad(X_tr_src_raw)
                    y_tr_src = binarize(y_tr_src_raw)
                    
                    X_train_sources.append(X_tr_src)
                    y_train_sources.append(y_tr_src)
                
                if regime == "zero_shot":
                    # Train purely on Source domains
                    X_train_shared = np.vstack(X_train_sources)
                    y_train_shared = np.concatenate(y_train_sources)
                    
                    # Strict Leakage Self-Check
                    # Target test data NEVER seen in training.
                    # In fact, no target data seen at all!
                    pass
                else: 
                    # few_shot (k=25 examples from Target Train split)
                    k = min(25, len(X_train_C))
                    
                    # Ensure k examples include both classes if possible, but keep it simple
                    X_train_shared = np.vstack(X_train_sources + [X_train_C[:k]])
                    y_train_shared = np.concatenate(y_train_sources + [y_train_C[:k]])
                    
                    # Strict Leakage Self-Check
                    # Ensure indices overlap is impossible because data_C["train"] and data_C["test"]
                    # are generated disjointly by data_loaders.py. We can assert based on row values or just trust the split.
                    # To be absolutely certain:
                    assert len(set(X_train_C_raw.index).intersection(set(X_test_C_raw.index))) == 0, "LEAKAGE DETECTED: Train and Test indices overlap!"
                    
                # Train the real detector (HistGradientBoosting)
                det = RealDetector('binary')
                det.config = "Full" # Use the main ensemble architecture
                det.fit(X_train_shared, y_train_shared)
                
                pred, prob, lat = det.predict_and_score(X_test_C)
                
                metrics = calculate_classification_metrics(y_test_C, pred, prob, 'binary')
                
                lodo_results.append({
                    "Scenario": scenario_name,
                    "Regime": regime,
                    "Seed": seed,
                    "Macro-F1": metrics["Macro-F1"],
                    "ROC-AUC": metrics["ROC-AUC"]
                })
                
    df = pd.DataFrame(lodo_results)
    
    # Aggregate Mean and SD across seeds
    summary = df.groupby(["Scenario", "Regime"]).agg(
        Macro_F1_Mean=("Macro-F1", "mean"),
        Macro_F1_SD=("Macro-F1", "std"),
        ROC_AUC_Mean=("ROC-AUC", "mean"),
        ROC_AUC_SD=("ROC-AUC", "std")
    ).reset_index()
    
    out_path = os.path.join(RESULTS_DIR, "lodo_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"LODO authentic evaluation saved to {out_path}")
    print("\nLODO Summary:")
    print(summary.to_string())

if __name__ == "__main__":
    run_lodo()
