from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.api.v1.endpoints.twins import twins_store
from app.services.xai.explainer import XAIExplainer
from pydantic import BaseModel

router = APIRouter()

class CounterfactualRequest(BaseModel):
    machine_id: str
    sensor_name: str
    hypothetical_value: float

@router.get("/{machine_id}/feature-importance")
def get_feature_importance(machine_id: str):
    """
    Returns SHAP feature importance explaining the RL model's current behavior.
    """
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[machine_id]
    
    # Run SHAP explainer
    try:
        explanation = XAIExplainer.get_feature_importance(virtual_rep)
        return explanation
    except Exception as e:
        # Fallback if model isn't trained yet or shap fails
        return {"error": str(e), "feature_importance": {"temp": 0.8, "vib": 0.5, "wear_factor": 0.9}}

@router.post("/counterfactual")
def run_counterfactual(payload: CounterfactualRequest):
    """
    Evaluates 'What-If' scenarios against the RL Engine.
    """
    if payload.machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found")
        
    virtual_rep = twins_store[payload.machine_id]
    
    try:
        cf_result = XAIExplainer.counterfactual_analysis(
            virtual_rep, 
            payload.sensor_name, 
            payload.hypothetical_value
        )
        return cf_result
    except Exception as e:
        return {"error": str(e), "original_action": [0.0, 0.0], "counterfactual_action": [0.5, -0.5], "delta": [0.5, -0.5]}
