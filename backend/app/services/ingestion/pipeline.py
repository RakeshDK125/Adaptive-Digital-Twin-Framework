import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class SensorPayload(BaseModel):
    machine_id: str
    timestamp: Optional[datetime] = None
    sensors: Dict[str, float]

    @field_validator('sensors')
    def check_empty_sensors(cls, v):
        if not v:
            raise ValueError("Sensors payload cannot be empty")
        return v

class PreprocessingPipeline:
    """
    Handles streaming preprocessing for incoming IoT telemetry.
    Maintains a rolling window state for noise reduction and outlier detection.
    """
    def __init__(self, window_size: int = 10, outlier_z_threshold: float = 3.0):
        self.window_size = window_size
        self.outlier_z_threshold = outlier_z_threshold
        # Stores history per machine per sensor: {machine_id: {sensor_name: [values]}}
        self.history_buffer: Dict[str, Dict[str, List[float]]] = {}
        # Stores last known good value for forward filling missing data
        self.last_known_values: Dict[str, Dict[str, float]] = {}

    def _update_buffer(self, machine_id: str, sensor: str, value: float):
        if machine_id not in self.history_buffer:
            self.history_buffer[machine_id] = {}
        if sensor not in self.history_buffer[machine_id]:
            self.history_buffer[machine_id][sensor] = []
            
        buffer = self.history_buffer[machine_id][sensor]
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)

    def _update_last_known(self, machine_id: str, sensor: str, value: float):
        if machine_id not in self.last_known_values:
            self.last_known_values[machine_id] = {}
        self.last_known_values[machine_id][sensor] = value

    def _get_last_known(self, machine_id: str, sensor: str) -> Optional[float]:
        return self.last_known_values.get(machine_id, {}).get(sensor)

    def process(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates, cleans, and extracts features from a single payload.
        """
        try:
            validated_data = SensorPayload(**raw_payload)
        except Exception as e:
            # In production, log validation errors
            raise ValueError(f"Invalid payload: {e}")

        machine_id = validated_data.machine_id
        timestamp = validated_data.timestamp or datetime.now()
        
        processed_sensors = {}
        features = {}

        for sensor, raw_value in validated_data.sensors.items():
            is_outlier = False
            processed_value = raw_value

            # 1. Missing Value Handling (if value is somehow NaN/None in a loose typing scenario)
            if raw_value is None or np.isnan(raw_value):
                last_val = self._get_last_known(machine_id, sensor)
                if last_val is not None:
                    processed_value = last_val
                else:
                    # No history, skip this sensor
                    continue

            # 2. Outlier Detection (Z-Score on historical buffer)
            history = self.history_buffer.get(machine_id, {}).get(sensor, [])
            if len(history) >= 3:
                mean = np.mean(history)
                std = np.std(history)
                if std > 0:
                    z_score = abs((processed_value - mean) / std)
                    if z_score > self.outlier_z_threshold:
                        is_outlier = True
                        # Optional: Clip outlier to threshold limit, or ignore. We will flag but apply EMA.
            
            # 3. Noise Removal (Exponential Moving Average smoothing)
            if history:
                alpha = 0.3
                ema = (alpha * processed_value) + ((1 - alpha) * history[-1])
                processed_value = ema

            # 4. Update buffers
            self._update_buffer(machine_id, sensor, processed_value)
            self._update_last_known(machine_id, sensor, processed_value)

            # 5. Feature Extraction (e.g., local variance or velocity)
            if len(history) >= 2:
                velocity = processed_value - history[-1]
                features[f"{sensor}_velocity"] = velocity
                features[f"{sensor}_rolling_std"] = np.std(history)

            processed_sensors[sensor] = {
                "raw_value": raw_value,
                "processed_value": processed_value,
                "is_outlier": is_outlier
            }

        return {
            "machine_id": machine_id,
            "timestamp": timestamp,
            "sensors": processed_sensors,
            "features": features
        }
