from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid
import copy

class DigitalAssetModel:
    """Base class for all digital twin entities."""
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now(timezone.utc)
        self.status = "INITIALIZED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }

class SensorModel:
    """Represents a physical sensor measuring a specific property."""
    def __init__(self, name: str, unit: str, min_threshold: float, max_threshold: float):
        self.name = name
        self.unit = unit
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.current_value: Optional[float] = None
        self.last_updated: Optional[datetime] = None
        self.is_anomalous: bool = False

    def update_value(self, value: float) -> None:
        self.current_value = value
        self.last_updated = datetime.now(timezone.utc)
        self.is_anomalous = value < self.min_threshold or value > self.max_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "current_value": self.current_value,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_anomalous": self.is_anomalous
        }

class MachineModel(DigitalAssetModel):
    """Represents an industrial machine comprised of multiple sensors."""
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.sensors: Dict[str, SensorModel] = {}
        # Operational parameters for simulation (e.g., efficiency, wear factor)
        self.parameters: Dict[str, float] = {"efficiency": 1.0, "wear": 0.0}

    def add_sensor(self, sensor: SensorModel) -> None:
        self.sensors[sensor.name] = sensor

    def get_sensor(self, name: str) -> Optional[SensorModel]:
        return self.sensors.get(name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "sensors": {name: sensor.to_dict() for name, sensor in self.sensors.items()},
            "parameters": self.parameters
        })
        return data

class EnvironmentModel(DigitalAssetModel):
    """Represents the ambient conditions affecting the machines."""
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.conditions: Dict[str, float] = {}

    def update_condition(self, name: str, value: float) -> None:
        self.conditions[name] = value

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["conditions"] = self.conditions
        return data

class VirtualRepresentation:
    """Aggregates a machine and its environment into a single virtual state."""
    def __init__(self, machine: MachineModel, environment: Optional[EnvironmentModel] = None):
        self.machine = machine
        self.environment = environment
        self.version = 1
        
    def clone_state(self) -> 'VirtualRepresentation':
        """Deep copy to capture a static state snapshot."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "machine": self.machine.to_dict(),
            "environment": self.environment.to_dict() if self.environment else None
        }
