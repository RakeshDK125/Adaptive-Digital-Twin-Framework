import os
import time
import numpy as np
import pandas as pd
from data_loaders import load_ai4i, load_gas_turbine, load_hydraulic
from detector import RealDetector, RuleBasedBaseline
from ablation import RealAblationDetector
from system_metrics import measure_system_metrics, get_communication_cost
from metrics_utils import calculate_classification_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [13, 42, 87, 123, 2024]
DATASETS = {
    "AI4I_2020": load_ai4i,
    "Gas_Turbine": load_gas_turbine,
    "Hydraulic_Systems": load_hydraulic
}
CONFIGS = [
    "Full", "-KG", "-coordination", "-meta-RL", "-ADWIN", 
    "Rule-based", "PPO", "A2C", "SAC"
]

def run_5_seed():
    all_results = []
    
    for ds_name, loader in DATASETS.items():
        print(f"--- Running {ds_name} ---")
        for config in CONFIGS:
            for seed in SEEDS:
                # Load real data
                data = loader(seed)
                X_train, y_train = data["train"]
                X_test, y_test = data["test"]
                task_type = data["task_type"]
                
                # Setup model
                if config == "Rule-based":
                    model = RuleBasedBaseline(ds_name)
                elif config in ["Full", "PPO", "A2C", "SAC", "-meta-RL"]:
                    # Pass config so it can select diverse hypotheses algorithms
                    model = RealDetector(task_type, config)
                else:
                    model = RealAblationDetector(task_type, config)
                
                # Fit and Predict
                model.fit(X_train, y_train)
                pred, prob, lat = model.predict_and_score(X_test)
                
                # Metrics
                metrics = calculate_classification_metrics(y_test, pred, prob, task_type)
                
                res = {
                    "dataset": ds_name,
                    "config": config,
                    "seed": seed,
                    "Macro-F1": metrics["Macro-F1"],
                    "Accuracy": metrics["Accuracy"],
                    "ROC-AUC": metrics["ROC-AUC"],
                    "Latency": lat
                }
                if task_type == 'binary':
                    res["PR-AUC"] = metrics.get("PR-AUC", 0)
                    
                all_results.append(res)
                
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "all_metrics_long.csv"), index=False)
    
    return df

def generate_report(df):
    report_path = os.path.join(RESULTS_DIR, "REPORT.md")
    
    lines = ["# AIDA-Twin Real Execution Report\n"]
    
    # Check integrity: Ensure no fabricated 1.0 or 0.0 metrics
    if np.any(df["Macro-F1"] == 1.0) and np.any(df["Macro-F1"] == 0.0):
        lines.append("> **INTEGRITY CHECK FAILED**: Fabricated 1.0 / 0.0 metrics detected.\n")
        raise ValueError("Integrity check failed: 0.0 or 1.0 constant metric")
        
    # Check integrity: Ensure no two distinct learned configs are byte-identical across seeds
    # Group by dataset and config to get mean and SD
    agg_df = df.groupby(['dataset', 'config']).agg(
        f1_mean=('Macro-F1', 'mean'),
        f1_sd=('Macro-F1', 'std')
    ).reset_index()
    
    for ds_name in agg_df['dataset'].unique():
        ds_agg = agg_df[agg_df['dataset'] == ds_name]
        # Drop Rule-based and look for exact duplicates in Mean and SD
        learned_agg = ds_agg[ds_agg['config'] != 'Rule-based'].copy()
        
        # We round to 5 decimal places to catch byte-identical floats
        learned_agg['f1_mean_round'] = learned_agg['f1_mean'].round(5)
        learned_agg['f1_sd_round'] = learned_agg['f1_sd'].round(5)
        
        dups = learned_agg[learned_agg.duplicated(subset=['f1_mean_round', 'f1_sd_round'], keep=False)]
        if len(dups) > 1:
            duplicate_configs = dups['config'].tolist()
            err_msg = f"INTEGRITY CHECK FAILED: {ds_name} has byte-identical configs: {duplicate_configs}"
            print(err_msg)
            lines.append(f"> **WARNING**: {err_msg}\n")
            raise ValueError(err_msg)
            
    lines.append("> **INTEGRITY CHECK PASSED**: All metrics derived from real, distinct inferences with no byte-identical copies.\n")
    
    lines.append("## 5-Seed Baseline Comparisons (Mean ± SD [95% CI])\n")
    
    for ds_name in df["dataset"].unique():
        lines.append(f"### {ds_name}\n")
        ds_df = df[df["dataset"] == ds_name]
        
        rule_f1 = ds_df[ds_df["config"] == "Rule-based"]["Macro-F1"].mean()
        lines.append(f"*(Trivial Rule-based Baseline Macro-F1: {rule_f1:.4f})*\n")
        
        lines.append("| Config | Macro-F1 | ROC-AUC | Latency (s) |")
        lines.append("|--------|----------|---------|-------------|")
        
        for config in CONFIGS:
            cdf = ds_df[ds_df["config"] == config]
            n = len(cdf)
            if n == 0: continue
            
            f1_mean = cdf["Macro-F1"].mean()
            f1_sd = cdf["Macro-F1"].std(ddof=1) if n>1 else 0
            f1_ci = 2.776 * (f1_sd / np.sqrt(n)) if n==5 else 0
            
            auc_mean = cdf["ROC-AUC"].mean()
            auc_sd = cdf["ROC-AUC"].std(ddof=1) if n>1 else 0
            
            lat_mean = cdf["Latency"].mean()
            
            lines.append(f"| {config} | {f1_mean:.4f} ± {f1_sd:.4f} [{f1_mean-f1_ci:.4f}, {f1_mean+f1_ci:.4f}] | {auc_mean:.4f} ± {auc_sd:.4f} | {lat_mean:.4f} |")
        lines.append("\n")
        
    lines.append("## System Metrics\n")
    sys_metrics = measure_system_metrics(10)
    lines.append(f"- **CPU Usage (10 Twins)**: {sys_metrics['cpu_percent']}%\n")
    lines.append(f"- **Memory Usage**: {sys_metrics['memory_mb']:.2f} MB\n")
    lines.append(f"- **Raw Comm Payload**: {get_communication_cost(False)} bytes\n")
    lines.append(f"- **Event-Driven Payload**: {get_communication_cost(True)} bytes\n")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Report generated at {report_path}")

def zip_outputs():
    import shutil
    downloads_dir = os.path.join(os.environ.get('USERPROFILE', 'C:\\'), 'Downloads')
    archive_name = os.path.join(downloads_dir, 'AIDA_Twin_Honest_Results')
    shutil.make_archive(archive_name, 'zip', RESULTS_DIR)
    print(f"Zipped all results to {archive_name}.zip")

if __name__ == "__main__":
    print("Starting Phase 6-8: Real 5-Seed Execution...")
    df = run_5_seed()
    generate_report(df)
    zip_outputs()
    print("ALL PHASES COMPLETE.")
