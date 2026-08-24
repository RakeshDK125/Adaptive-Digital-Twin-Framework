import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from app.domain.twin.models import VirtualRepresentation

class TwinSnapshot:
    """A frozen state capture of a twin at a specific point in time."""
    def __init__(self, virtual_rep: VirtualRepresentation):
        self.snapshot_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.state_data = virtual_rep.to_dict()

class TwinVersioning:
    """Tracks structural or parameter changes to a twin."""
    @staticmethod
    def increment_version(virtual_rep: VirtualRepresentation) -> None:
        virtual_rep.version += 1

class TwinComparison:
    """Compares two snapshots to identify drift and structural changes."""
    @staticmethod
    def compare(snapshot_a: TwinSnapshot, snapshot_b: TwinSnapshot) -> Dict[str, Any]:
        """
        Returns a dictionary of deltas between snapshot_a (older) and snapshot_b (newer).
        """
        machine_a = snapshot_a.state_data["machine"]
        machine_b = snapshot_b.state_data["machine"]
        
        deltas = {
            "version_diff": machine_b.get("version", 1) - machine_a.get("version", 1),
            "sensor_drift": {},
            "parameter_drift": {}
        }
        
        # Sensor value drift
        sensors_a = machine_a.get("sensors", {})
        sensors_b = machine_b.get("sensors", {})
        
        for name, data_b in sensors_b.items():
            if name in sensors_a:
                val_a = sensors_a[name].get("current_value")
                val_b = data_b.get("current_value")
                if val_a is not None and val_b is not None:
                    deltas["sensor_drift"][name] = val_b - val_a
                    
        # Parameter drift
        params_a = machine_a.get("parameters", {})
        params_b = machine_b.get("parameters", {})
        for param, val_b in params_b.items():
            if param in params_a:
                deltas["parameter_drift"][param] = val_b - params_a[param]
                
        return deltas

class HistoricalTwin:
    """Maintains a timeline of snapshots for a specific machine."""
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.timeline: List[TwinSnapshot] = []
        
    def add_snapshot(self, snapshot: TwinSnapshot) -> None:
        self.timeline.append(snapshot)
        # Sort chronologically just in case
        self.timeline.sort(key=lambda s: s.timestamp)
        
    def get_snapshots_in_range(self, start: datetime, end: datetime) -> List[TwinSnapshot]:
        return [s for s in self.timeline if start <= s.timestamp <= end]

class TwinReplay:
    """Replays a sequence of snapshots."""
    def __init__(self, history: HistoricalTwin):
        self.history = history
        
    def replay(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """
        Yields the states chronologically for the given time window.
        """
        snapshots = self.history.get_snapshots_in_range(start, end)
        return [s.state_data for s in snapshots]
