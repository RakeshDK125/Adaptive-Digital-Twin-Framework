import pytest
from datetime import datetime, timezone
from app.domain.twin.models import MachineModel, SensorModel, VirtualRepresentation
from app.services.twin.engine import StateSynchronization, SimulationEngine, AdaptiveTwinUpdateEngine, TwinHealthMonitoring
from app.services.twin.history import TwinSnapshot, HistoricalTwin, TwinComparison, TwinReplay

@pytest.fixture
def sample_virtual_rep():
    machine = MachineModel(name="Test CNC")
    sensor = SensorModel(name="temperature", unit="C", min_threshold=10.0, max_threshold=80.0)
    machine.add_sensor(sensor)
    return VirtualRepresentation(machine)

def test_sensor_bounds(sample_virtual_rep):
    sensor = sample_virtual_rep.machine.get_sensor("temperature")
    sensor.update_value(50.0)
    assert sensor.is_anomalous is False
    
    sensor.update_value(90.0)
    assert sensor.is_anomalous is True

def test_state_synchronization(sample_virtual_rep):
    payload = {"temperature": 45.5}
    StateSynchronization.sync_telemetry(sample_virtual_rep, payload)
    
    sensor = sample_virtual_rep.machine.get_sensor("temperature")
    assert sensor.current_value == 45.5
    assert sample_virtual_rep.machine.status == "SYNCED"

def test_simulation_engine(sample_virtual_rep):
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 20.0})
    
    sim_rep = SimulationEngine.step_forward(sample_virtual_rep, steps=5)
    
    # Original should be unchanged
    assert sample_virtual_rep.machine.get_sensor("temperature").current_value == 20.0
    
    # Simulated value should have increased
    sim_sensor = sim_rep.machine.get_sensor("temperature")
    assert sim_sensor.current_value > 20.0
    
    # Wear factor should have increased
    assert sim_rep.machine.parameters["wear"] > 0.0

def test_adaptive_update_engine(sample_virtual_rep):
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 25.0})
    sim_rep = SimulationEngine.step_forward(sample_virtual_rep, steps=1)
    
    # Mock real drift
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 30.0})
    
    drift = AdaptiveTwinUpdateEngine.calculate_drift_and_adapt(sample_virtual_rep, sim_rep)
    
    assert "temperature" in drift
    assert sample_virtual_rep.version == 2

def test_health_monitoring(sample_virtual_rep):
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 50.0})
    score = TwinHealthMonitoring.calculate_health_score(sample_virtual_rep)
    assert score == 100.0
    
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 90.0})
    score_bad = TwinHealthMonitoring.calculate_health_score(sample_virtual_rep)
    assert score_bad < 100.0

def test_historical_tracking(sample_virtual_rep):
    history = HistoricalTwin(sample_virtual_rep.machine.id)
    
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 20.0})
    snap1 = TwinSnapshot(sample_virtual_rep)
    history.add_snapshot(snap1)
    
    StateSynchronization.sync_telemetry(sample_virtual_rep, {"temperature": 40.0})
    snap2 = TwinSnapshot(sample_virtual_rep)
    history.add_snapshot(snap2)
    
    deltas = TwinComparison.compare(snap1, snap2)
    assert deltas["sensor_drift"]["temperature"] == 20.0
    
    replay = TwinReplay(history)
    start_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2100, 1, 1, tzinfo=timezone.utc)
    
    frames = replay.replay(start_time, end_time)
    assert len(frames) == 2
