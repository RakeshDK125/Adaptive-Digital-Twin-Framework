import time
import psutil
import os
import random

def measure_system_metrics(num_twins=1):
    """
    Measures real CPU, Memory, and Communication payload sizes.
    """
    process = psutil.Process(os.getpid())
    
    # Warmup CPU measurement
    psutil.cpu_percent(interval=None)
    
    start_time = time.perf_counter()
    
    # Simulate workload of N twins
    dummy = []
    for _ in range(num_twins * 1000):
        dummy.append(random.random() ** 2)
        
    cpu = psutil.cpu_percent(interval=0.1)
    mem_mb = process.memory_info().rss / (1024 * 1024)
    resp_time = time.perf_counter() - start_time
    
    return {
        "num_twins": num_twins,
        "cpu_percent": cpu,
        "memory_mb": mem_mb,
        "response_time_s": resp_time
    }

def get_communication_cost(event_driven=True):
    """
    Returns actual byte sizes for the data payload.
    """
    import json
    # A standard payload
    payload = {
        "sensor_1": 45.2,
        "sensor_2": 12.1,
        "status": "active"
    }
    
    raw_size = len(json.dumps(payload).encode('utf-8'))
    
    if event_driven:
        # Event driven only sends the delta or flag
        event_payload = {"status": "active"}
        return len(json.dumps(event_payload).encode('utf-8'))
    return raw_size

if __name__ == "__main__":
    print(measure_system_metrics(10))
    print(f"Raw Comm: {get_communication_cost(False)} bytes")
    print(f"Event Comm: {get_communication_cost(True)} bytes")
