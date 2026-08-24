from locust import HttpUser, task, between
import random

class IoTSensorSim(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def simulate_telemetry(self):
        machine_id = "test_machine"
        payload = {
            "machine_id": machine_id,
            "timestamp": "2026-08-01T12:00:00Z",
            "sensors": {
                "temp": random.uniform(20.0, 100.0),
                "vib": random.uniform(0.1, 10.0),
                "rpm": random.uniform(1000, 3000)
            }
        }
        
        self.client.post("/api/v1/twins/ingest", json=payload)
