import random
from typing import Dict, Any, List
from app.domain.twin.models import VirtualRepresentation, MachineModel

class StateSynchronization:
    """Synchronizes physical telemetry with the virtual twin."""
    @staticmethod
    def sync_telemetry(virtual_rep: VirtualRepresentation, payload: Dict[str, float]) -> None:
        """
        Ingests a dictionary of {sensor_name: value} and updates the virtual representation.
        """
        machine = virtual_rep.machine
        for sensor_name, value in payload.items():
            sensor = machine.get_sensor(sensor_name)
            if sensor:
                sensor.update_value(value)
                
        machine.status = "SYNCED"

class SimulationEngine:
    """Predicts the next state based on current parameters and environment."""
    @staticmethod
    def step_forward(virtual_rep: VirtualRepresentation, steps: int = 1) -> VirtualRepresentation:
        """
        Simulates physics/logic forward in time without altering the baseline true state.
        Returns a new simulated VirtualRepresentation.
        """
        sim_rep = virtual_rep.clone_state()
        machine = sim_rep.machine
        env = sim_rep.environment
        
        # Simple example simulation logic: temperature increases with wear, vibration adds noise
        ambient_temp = env.conditions.get("ambient_temperature", 20.0) if env else 20.0
        wear_factor = machine.parameters.get("wear", 0.0)
        
        for _ in range(steps):
            for sensor_name, sensor in machine.sensors.items():
                if sensor.current_value is not None:
                    # Apply a mock physics step
                    if "temp" in sensor_name.lower():
                        # Temperature tends towards ambient + operational heat (which increases with wear)
                        heat_generated = 5.0 * (1.0 + wear_factor)
                        sensor.update_value(sensor.current_value + (ambient_temp + heat_generated - sensor.current_value) * 0.1)
                    elif "vib" in sensor_name.lower():
                        # Vibration increases with wear + random noise
                        noise = random.uniform(-0.5, 0.5)
                        sensor.update_value(sensor.current_value + (wear_factor * 0.1) + noise)
            
            # Increment wear slightly over time
            machine.parameters["wear"] += 0.001

        return sim_rep

class AdaptiveTwinUpdateEngine:
    """Adapts simulation parameters by comparing real vs simulated states."""
    @staticmethod
    def calculate_drift_and_adapt(true_rep: VirtualRepresentation, sim_rep: VirtualRepresentation) -> Dict[str, float]:
        """
        Compares the real state against what the simulation predicted.
        Adjusts the true machine's parameters (e.g., wear) to minimize future drift.
        Returns the calculated drift metrics.
        """
        drifts = {}
        total_drift = 0.0
        
        for name, true_sensor in true_rep.machine.sensors.items():
            sim_sensor = sim_rep.machine.get_sensor(name)
            if true_sensor.current_value is not None and sim_sensor and sim_sensor.current_value is not None:
                drift = true_sensor.current_value - sim_sensor.current_value
                drifts[name] = drift
                total_drift += abs(drift)
        
        # Naive adaptation: if average simulated vibration is lower than true, increase wear factor
        if "vibration" in drifts and drifts["vibration"] > 0.5:
             true_rep.machine.parameters["wear"] += 0.05
             
        # Increment version as parameters have adapted
        true_rep.version += 1
        
        return drifts

class TwinHealthMonitoring:
    """Evaluates the health of the digital twin."""
    @staticmethod
    def calculate_health_score(virtual_rep: VirtualRepresentation) -> float:
        """
        Returns a health score between 0.0 (Failed) and 100.0 (Perfect).
        Penalizes anomalous sensors and high wear.
        """
        machine = virtual_rep.machine
        score = 100.0
        
        if not machine.sensors:
            return score
            
        anomaly_penalty = 100.0 / len(machine.sensors)
        for sensor in machine.sensors.values():
            if sensor.is_anomalous:
                score -= anomaly_penalty
                
        # Deduct based on wear factor
        wear = machine.parameters.get("wear", 0.0)
        score -= min(wear * 50.0, score) # Max penalty for wear is 50 points or current score
        
        return max(0.0, round(score, 2))
