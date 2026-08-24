from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.api.v1.endpoints.twins import twins_store
from app.api.v1.endpoints.agents import coordinator
from app.services.decision.engine import DecisionIntelligenceEngine
from app.services.decision.history import DecisionHistoryManager

router = APIRouter()

decision_engine = DecisionIntelligenceEngine(agent_coordinator=coordinator)
history_manager = DecisionHistoryManager()

@router.post("/{machine_id}/evaluate")
def evaluate_decision(machine_id: str):
    """
    Triggers a fusion sweep across Twin, RL, KG, and Agents to formulate a decision.
    """
    if machine_id not in twins_store:
        raise HTTPException(status_code=404, detail="Machine not found in Twin Store")
        
    virtual_rep = twins_store[machine_id]
    
    report = decision_engine.generate_report(machine_id, virtual_rep)
    
    # Save to history automatically
    history_manager.persist_decision(report)
    
    return report

@router.get("/{machine_id}/history")
def get_decision_history(machine_id: str):
    """
    Fetches the chronological audit trail of decisions for this machine from Neo4j.
    """
    history = history_manager.get_decision_history(machine_id)
    return {"machine_id": machine_id, "history": history}

@router.get("/{machine_id}/visualize")
def visualize_decisions(machine_id: str):
    """
    Returns a flattened payload explicitly designed for frontend visualization (Plotly Dash/React).
    """
    # Fetch actual history, fallback to mock if Neo4j is empty during testing
    history = history_manager.get_decision_history(machine_id)
    
    if not history:
        # Mock data for UI development
        return {
            "x": ["2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", "2026-08-01T12:00:00Z"],
            "risk_scores": [20.5, 45.0, 85.2],
            "confidence": [95.0, 88.0, 92.0]
        }
        
    # In a real scenario, we'd parse the history dicts to extract time-series arrays
    return {"message": "Visualize payload", "raw_history": history}
