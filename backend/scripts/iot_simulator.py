import time
import json
import random
import requests
import threading

REST_URL = "http://localhost:8000/api/v1/twins"

def generate_sensor_data(base_temp=50.0, base_vib=2.0):
    """Generates synthetic data with noise and occasional outliers."""
    temp = base_temp + random.uniform(-2, 2)
    vib = base_vib + random.uniform(-0.1, 0.1)
    
    # 5% chance of outlier
    if random.random() < 0.05:
        temp += random.choice([-30, 30])
        print(">> Injected Outlier <<")
        
    # 2% chance of missing data (None)
    if random.random() < 0.02:
        vib = None
        print(">> Injected Missing Value <<")
        
    return {"temperature": temp, "vibration": vib}

def simulate_rest(machine_id: str, steps: int = 50):
    for i in range(steps):
        data = generate_sensor_data()
        try:
            res = requests.post(f"{REST_URL}/machines/{machine_id}/telemetry", json=data)
            print(f"REST Ingest [{i}]: {res.status_code}")
        except Exception as e:
            print(f"REST Error: {e}")
        time.sleep(1)

def setup_machine() -> str:
    payload = {
        "name": "Simulated Ingestion Turbine",
        "sensors": [
            {"name": "temperature", "unit": "C", "min_threshold": 0.0, "max_threshold": 100.0},
            {"name": "vibration", "unit": "mm/s", "min_threshold": 0.0, "max_threshold": 10.0}
        ]
    }
    res = requests.post(f"{REST_URL}/machines", json=payload)
    if res.status_code == 201:
        return res.json()["id"]
    return "test_machine"

if __name__ == "__main__":
    print("Starting IoT Simulator...")
    machine_id = setup_machine()
    print(f"Targeting Machine ID: {machine_id}")
    
    # In a real scenario, we'd spin up MQTT/Kafka publishers here too.
    # For now, we simulate the REST stream which feeds the exact same PreprocessingPipeline
    t1 = threading.Thread(target=simulate_rest, args=(machine_id, 20))
    t1.start()
    t1.join()
    
    print("Simulation Complete.")
