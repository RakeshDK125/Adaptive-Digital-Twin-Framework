from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

from app.domain.twin.models import MachineModel, SensorModel, VirtualRepresentation, EnvironmentModel
from app.services.twin.engine import StateSynchronization, SimulationEngine, AdaptiveTwinUpdateEngine, TwinHealthMonitoring
from app.services.twin.history import TwinSnapshot, HistoricalTwin, TwinComparison, TwinReplay

router = APIRouter()

# In-memory store for multiple digital twins and their histories
# In a production setting, this would be backed by Redis or PostgreSQL/Neo4j
twins_store: Dict[str, VirtualRepresentation] = {}
history_store: Dict[str, HistoricalTwin] = {}

class SensorCreate(BaseModel):
    name: str
    unit: str
    min_threshold: float
    max_threshold: float

class MachineCreate(BaseModel):
    name: str
    description: str = ""
    sensors: List[SensorCreate]

@router.post("/machines", response_model=Dict[str, Any], status_code=201)
def create_machine(payload: MachineCreate):
    machine = MachineModel(name=payload.name, description=payload.description)
    for s in payload.sensors:
        sensor = SensorModel(s.name, s.unit, s.min_threshold, s.max_threshold)
        machine.add_sensor(sensor)
        
    env = EnvironmentModel(name=f"Env for {payload.name}")
    virtual_rep = VirtualRepresentation(machine, env)
    
    twins_store[machine.id] = virtual_rep
    history_store[machine.id] = HistoricalTwin(machine.id)
    
    return {"message": "Machine created", "id": machine.id, "state": virtual_rep.to_dict()}

@router.get("/machines/{machine_id}/state")
def get_machine_state(machine_id: str):
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
    return twins_store[machine_id].to_dict()

from app.services.ingestion.connectors import IngestionManager

def _sync_callback(machine_id: str, processed_payload: Dict[str, float]):
    if machine_id in twins_store:
        virtual_rep = twins_store[machine_id]
        StateSynchronization.sync_telemetry(virtual_rep, processed_payload)

# Initialize the manager globally for the REST endpoint
rest_ingestion_manager = IngestionManager(twin_sync_callback=_sync_callback)

@router.post("/machines/{machine_id}/telemetry")
def ingest_telemetry(machine_id: str, payload: Dict[str, float] = Body(...)):
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    # Route through the preprocessing pipeline instead of direct sync
    raw_payload = {
        "machine_id": machine_id,
        "sensors": payload
    }
    rest_ingestion_manager.handle_payload(raw_payload)
    
    return {"message": "Telemetry ingested and synced successfully", "state": twins_store[machine_id].to_dict()}

@router.post("/machines/{machine_id}/simulate")
def simulate_forward(machine_id: str, steps: int = 1):
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[machine_id]
    sim_rep = SimulationEngine.step_forward(virtual_rep, steps)
    
    # Calculate drift if we want to adapt immediately (for demonstration)
    drift = AdaptiveTwinUpdateEngine.calculate_drift_and_adapt(virtual_rep, sim_rep)
    
    return {
        "message": f"Simulated {steps} steps", 
        "simulated_state": sim_rep.to_dict(),
        "drift_adapted": drift
    }

@router.get("/machines/{machine_id}/health")
def get_health_score(machine_id: str):
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[machine_id]
    score = TwinHealthMonitoring.calculate_health_score(virtual_rep)
    
    return {"machine_id": machine_id, "health_score": score}

@router.post("/machines/{machine_id}/snapshots")
def create_snapshot(machine_id: str):
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[machine_id]
    snapshot = TwinSnapshot(virtual_rep)
    
    history_store[machine_id].add_snapshot(snapshot)
    
    return {"message": "Snapshot created", "snapshot_id": snapshot.snapshot_id}

@router.get("/machines/{machine_id}/replay")
def replay_history(machine_id: str, start: datetime, end: datetime):
    if machine_id not in history_store:
        raise HTTPException(status_code=404, detail="Machine history not found")
        
    replay_engine = TwinReplay(history_store[machine_id])
    states = replay_engine.replay(start, end)
    
    return {"machine_id": machine_id, "frames": len(states), "states": states}
