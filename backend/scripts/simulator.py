import random
import time
from typing import Generator

class IndustrialEventSimulator:
    """
    Deterministically simulates thousands of hours of industrial machine run-time
    with synthetic noise, wear, and catastrophic fault injection.
    """
    def __init__(self, machine_id="SIM-01"):
        self.machine_id = machine_id
        self.wear = 0.0
        self.temp_base = 60.0
        self.vib_base = 1.0
        
    def stream_telemetry(self, timesteps: int) -> Generator[dict, None, None]:
        for t in range(timesteps):
            # Progressive degradation
            self.wear += random.uniform(0.0001, 0.001)
            
            # Synthetic noise
            temp = self.temp_base + (self.wear * 20.0) + random.gauss(0, 2)
            vib = self.vib_base + (self.wear * 5.0) + random.gauss(0, 0.5)
            
            # Catastrophic fault injection (e.g. concept drift)
            if t == timesteps // 2:
                self.temp_base += 30.0 # Sudden overheating
                
            yield {
                "timestamp": t,
                "machine_id": self.machine_id,
                "temp": temp,
                "vib": vib,
                "wear": self.wear
            }

if __name__ == "__main__":
    sim = IndustrialEventSimulator()
    for t in sim.stream_telemetry(10):
        print(t)
