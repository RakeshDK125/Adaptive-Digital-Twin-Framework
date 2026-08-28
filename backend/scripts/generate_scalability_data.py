import os
import time
import pandas as pd
import numpy as np
from five_seed_runs import create_env
from app.domain.twin.models import SensorModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'experiments_run')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_large_mock_dataset(num_sensors):
    np.random.seed(42)
    data = {}
    for i in range(num_sensors):
        data[f'Sensor_{i}'] = np.random.normal(100, 10, 100)
    return pd.DataFrame(data)

def main():
    print("Generating real scalability data...")
    sensor_counts = [10, 50, 100, 200, 500]
    results = []
    
    for count in sensor_counts:
        print(f"Testing scalability for {count} sensors...")
        df = generate_large_mock_dataset(count)
        env = create_env(df, seed=42)
        
        env.reset()
        
        # Warmup
        for _ in range(5):
            env.step(env.action_space.sample())
            
        # Measure
        start = time.perf_counter()
        for _ in range(50):
            env.step(env.action_space.sample())
        end = time.perf_counter()
        
        # Average step latency in milliseconds
        avg_latency_ms = ((end - start) / 50.0) * 1000.0
        
        results.append({
            "Number of Sensors": count,
            "Latency per Step (ms)": avg_latency_ms
        })
        
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(OUTPUT_DIR, 'scalability_results.csv'), index=False)
    print("Scalability data saved.")

if __name__ == "__main__":
    main()
