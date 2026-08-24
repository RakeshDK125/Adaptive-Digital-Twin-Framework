import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
from app.domain.twin.models import VirtualRepresentation
from app.services.twin.engine import SimulationEngine, TwinHealthMonitoring

class DigitalTwinEnv(gym.Env):
    """
    Custom Environment that maps the Digital Twin Virtual Representation 
    to a Gymnasium Interface for Reinforcement Learning.
    """
    def __init__(self, virtual_rep: VirtualRepresentation, max_steps: int = 1000):
        super(DigitalTwinEnv, self).__init__()
        
        self.base_virtual_rep = virtual_rep
        self.current_rep = virtual_rep.clone_state()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Determine sensors to build the observation space
        self.sensor_keys = sorted(list(self.base_virtual_rep.machine.sensors.keys()))
        num_sensors = len(self.sensor_keys)
        
        # State Space: Normalized sensor values + current wear factor
        # [sensor_1, sensor_2, ..., sensor_n, wear]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_sensors + 1,), dtype=np.float32
        )
        
        # Action Space: Continuous control (e.g., adjust cooling, adjust speed)
        # Let's assume 2 continuous actions [-1.0, 1.0] for general machine control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        obs = []
        for key in self.sensor_keys:
            sensor = self.current_rep.machine.sensors[key]
            # Naive normalization: (val - min) / (max - min) * 2 - 1
            val = sensor.current_value if sensor.current_value is not None else 0.0
            range_span = sensor.max_threshold - sensor.min_threshold
            if range_span == 0: range_span = 1.0
            
            norm_val = 2.0 * ((val - sensor.min_threshold) / range_span) - 1.0
            obs.append(np.clip(norm_val, -1.0, 1.0))
            
        wear = self.current_rep.machine.parameters.get("wear", 0.0)
        obs.append(np.clip(wear, 0.0, 1.0)) # wear is assumed 0-1
        
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_rep = self.base_virtual_rep.clone_state()
        
        # Inject some slight randomness into initial state if needed
        for sensor in self.current_rep.machine.sensors.values():
            if sensor.current_value is not None:
                sensor.current_value += self.np_random.uniform(-0.5, 0.5)
                
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Apply action to environment conditions before simulation step
        # Action 0: Adjust ambient temp (e.g., cooling flow)
        # Action 1: Adjust machine speed/stress (affects wear generation)
        if self.current_rep.environment:
            current_temp = self.current_rep.environment.conditions.get("ambient_temperature", 20.0)
            self.current_rep.environment.update_condition("ambient_temperature", current_temp + action[0] * 5.0)
            
        # Temporarily store original wear to apply speed action
        original_wear = self.current_rep.machine.parameters.get("wear", 0.0)
        
        # Run twin simulation step forward
        self.current_rep = SimulationEngine.step_forward(self.current_rep, steps=1)
        
        # Apply speed action to wear generated in that step
        new_wear = self.current_rep.machine.parameters.get("wear", 0.0)
        wear_delta = new_wear - original_wear
        adjusted_wear = original_wear + wear_delta * (1.0 + action[1])
        self.current_rep.machine.parameters["wear"] = max(0.0, adjusted_wear)
        
        # Calculate Adaptive Reward
        health_score = TwinHealthMonitoring.calculate_health_score(self.current_rep)
        
        # Reward is health score normalized to [-1, 1], minus penalties for aggressive actions
        base_reward = (health_score / 50.0) - 1.0 
        action_penalty = -0.1 * np.sum(np.square(action))
        
        reward = base_reward + action_penalty
        
        # Terminal condition: Health drops too low or max steps reached
        terminated = health_score < 20.0
        truncated = self.current_step >= self.max_steps
        
        info = {
            "health_score": health_score,
            "wear": self.current_rep.machine.parameters["wear"]
        }
        
        return self._get_observation(), reward, terminated, truncated, info
