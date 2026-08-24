import requests
import time
import random
from pprint import pprint

BASE_URL = "http://localhost:8000/api/v1/twins"

def generate_example_data():
    print("--- 1. Creating Machine ---")
    payload = {
        "name": "Industrial Turbine X-1",
        "description": "Main power generation turbine",
        "sensors": [
            {"name": "temperature", "unit": "C", "min_threshold": 20.0, "max_threshold": 120.0},
            {"name": "vibration", "unit": "mm/s", "min_threshold": 0.0, "max_threshold": 5.0},
            {"name": "pressure", "unit": "bar", "min_threshold": 1.0, "max_threshold": 10.0}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/machines", json=payload)
    if response.status_code != 201:
        print("Failed to create machine. Is the server running?")
        print(response.text)
        return
        
    data = response.json()
    machine_id = data["id"]
    print(f"Created Machine ID: {machine_id}")
    pprint(data["state"])
    print("\n")
    
    print("--- 2. Simulating Sensor Telemetry ---")
    for i in range(5):
        telemetry = {
            "temperature": random.uniform(60.0, 110.0),
            "vibration": random.uniform(1.0, 4.5),
            "pressure": random.uniform(5.0, 8.0)
        }
        res = requests.post(f"{BASE_URL}/machines/{machine_id}/telemetry", json=telemetry)
        print(f"Sync {i+1}: Status {res.status_code}")
        time.sleep(1)
        
    print("\n--- 3. Taking a Snapshot ---")
    res = requests.post(f"{BASE_URL}/machines/{machine_id}/snapshots")
    print(res.json())
    
    print("\n--- 4. Running Simulation Forward ---")
    res = requests.post(f"{BASE_URL}/machines/{machine_id}/simulate?steps=10")
    print("Simulation Output (Drift & State):")
    pprint(res.json())
    
    print("\n--- 5. Checking Machine Health ---")
    res = requests.get(f"{BASE_URL}/machines/{machine_id}/health")
    print(res.json())

if __name__ == "__main__":
    generate_example_data()
