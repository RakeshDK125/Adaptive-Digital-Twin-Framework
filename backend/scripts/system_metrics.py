import time
import psutil
import os
import numpy as np
import pandas as pd
import json
import platform
import concurrent.futures
from data_loaders import load_ai4i
from detector import RealDetector

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def measure_scalability():
    print("Measuring Real System Scalability...")
    
    # Load real data and train a single model to use for inference
    data = load_ai4i(42)
    X_train, y_train = data['train']
    X_test, _ = data['test']
    
    # We use a relatively quick model, but real enough
    det = RealDetector('binary')
    det.config = "Full"
    det.fit(X_train, y_train)
    
    # We'll run inference on a chunk of rows to simulate a standard twin processing interval
    X_sample = X_test.iloc[:500] 
    
    def inference_task():
        start = time.perf_counter()
        det.predict_and_score(X_sample)
        return time.perf_counter() - start

    results = []
    
    # Measure baseline
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # warmup
    
    for N in [1, 10, 50, 100]:
        print(f"Testing N={N} concurrent twins...")
        
        N_results = {"cpu": [], "rss": [], "resp": []}
        
        # 5 repeats
        for _ in range(5):
            # Record CPU before
            process.cpu_percent(interval=None)
            
            # Execute N twins concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(N, 100)) as executor:
                futures = [executor.submit(inference_task) for _ in range(N)]
                
                # Check CPU during execution (approximation)
                cpu_util = process.cpu_percent(interval=0.1)
                
                concurrent.futures.wait(futures)
                
                times = [f.result() for f in futures if not f.exception()]
                
            mem_mb = process.memory_info().rss / (1024 * 1024)
            
            # Since threads share CPU and we wait, the cpu_percent is for the interval
            N_results["cpu"].append(cpu_util)
            N_results["rss"].append(mem_mb)
            if times:
                N_results["resp"].append(np.mean(times) * 1000) # ms
                
        results.append({
            "Num_Twins": N,
            "CPU_Percent_Mean": np.mean(N_results["cpu"]),
            "CPU_Percent_SD": np.std(N_results["cpu"]),
            "Memory_MB_Mean": np.mean(N_results["rss"]),
            "Memory_MB_SD": np.std(N_results["rss"]),
            "Resp_Time_ms_Mean": np.mean(N_results["resp"]),
            "Resp_Time_ms_SD": np.std(N_results["resp"]),
        })
        
    df = pd.DataFrame(results)
    return df

def measure_communication(data_slice):
    """
    Measures the real payload size of a JSON serialized sensor update.
    """
    print("Measuring Real Communication Payloads...")
    
    raw_bytes = []
    event_bytes = []
    
    # Compute threshold for event-driven
    thresholds = data_slice.quantile(0.75)
    
    events_triggered = 0
    total_steps = len(data_slice)
    
    for i in range(total_steps):
        row = data_slice.iloc[i].to_dict()
        
        # 1. Raw Periodic: serialize everything
        raw_payload = json.dumps(row).encode('utf-8')
        raw_bytes.append(len(raw_payload))
        
        # 2. Event Driven: only serialize if a feature exceeds 75th percentile
        triggered = {}
        for col, val in row.items():
            if val > thresholds[col]:
                triggered[col] = val
                
        if triggered:
            events_triggered += 1
            event_payload = json.dumps(triggered).encode('utf-8')
            event_bytes.append(len(event_payload))
            
    mean_raw = np.mean(raw_bytes)
    # Mean event size across ALL steps (0 bytes if not triggered)
    total_event_bytes = sum(event_bytes)
    mean_event = total_event_bytes / total_steps
    
    reduction = ((mean_raw - mean_event) / mean_raw) * 100 if mean_raw > 0 else 0
    event_rate = events_triggered / total_steps
    
    res = {
        "Raw_Bytes_Per_Step": mean_raw,
        "Event_Bytes_Per_Step": mean_event,
        "Reduction_Percent": reduction,
        "Event_Rate": event_rate
    }
    return pd.DataFrame([res])

if __name__ == "__main__":
    # Print Hardware Stats
    hw_info = {
        "CPU": platform.processor(),
        "Cores": os.cpu_count(),
        "OS": platform.system() + " " + platform.release(),
        "RAM_GB": round(psutil.virtual_memory().total / (1024**3), 2),
        "Python": platform.python_version()
    }
    print("Hardware:", hw_info)
    
    # 1. Scalability
    scale_df = measure_scalability()
    
    # 2. Communication
    # Use real gas turbine data for sizing since it has ~11 physical sensors
    from data_loaders import load_gas_turbine
    gt_data = load_gas_turbine(42)
    X_test_gt, _ = gt_data['test']
    comm_df = measure_communication(X_test_gt.iloc[:1000]) # use 1000 rows
    
    # Combine and save
    out_path = os.path.join(RESULTS_DIR, "system_metrics.csv")
    
    with open(out_path, 'w') as f:
        f.write("# HARDWARE\n")
        json.dump(hw_info, f)
        f.write("\n\n# SCALABILITY\n")
        scale_df.to_csv(f, index=False)
        f.write("\n# COMMUNICATION\n")
        comm_df.to_csv(f, index=False)
        
    print(f"System metrics saved to {out_path}")
