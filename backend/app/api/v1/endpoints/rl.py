from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os

from app.rl.manager import RLManager, MODEL_DIR
from app.api.v1.endpoints.twins import twins_store

router = APIRouter()

class TrainRequest(BaseModel):
    machine_id: str
    algo: str = "PPO"
    timesteps: int = 10000
    model_name: str = "twin_agent"

class InferRequest(BaseModel):
    machine_id: str
    algo: str = "PPO"
    model_name: str

@router.post("/train")
def train_model(payload: TrainRequest, background_tasks: BackgroundTasks):
    if payload.machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[payload.machine_id]
    
    # Run training in background so we don't block the API
    def _train_task():
        try:
            RLManager.train_model(virtual_rep, payload.algo, payload.timesteps, payload.model_name)
        except Exception as e:
            print(f"Training failed: {e}")
            
    background_tasks.add_task(_train_task)
    return {"message": f"Training {payload.algo} model started in background"}

@router.get("/models")
def list_models():
    if not os.path.exists(MODEL_DIR):
        return {"models": []}
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip")]
    return {"models": files}

@router.post("/infer")
def get_optimal_action(payload: InferRequest):
    if payload.machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    model_path = os.path.join(MODEL_DIR, f"{payload.model_name}.zip")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    virtual_rep = twins_store[payload.machine_id]
    
    action = RLManager.get_action(virtual_rep, payload.algo, model_path)
    return {"machine_id": payload.machine_id, "optimal_action": action}

@router.post("/online-finetune")
def online_finetune(payload: TrainRequest, background_tasks: BackgroundTasks):
    if payload.machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    model_path = os.path.join(MODEL_DIR, f"{payload.model_name}.zip")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    virtual_rep = twins_store[payload.machine_id]
    
    def _finetune_task():
        RLManager.online_finetune(virtual_rep, payload.algo, model_path, payload.timesteps)
        
    background_tasks.add_task(_finetune_task)
    return {"message": "Online fine-tuning started"}
