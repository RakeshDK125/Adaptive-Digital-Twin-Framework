from typing import Dict, Any, List
from datetime import datetime, timezone
import uuid
import os

from app.services.twin.engine import TwinHealthMonitoring, SimulationEngine
from app.rl.manager import RLManager, MODEL_DIR
from app.services.kg.graph_manager import GraphManager
from app.domain.twin.models import VirtualRepresentation

class DecisionIntelligenceEngine:
    """
    Fuses outputs from Digital Twin, RL, Knowledge Graph, and Agentic AI
    to generate a comprehensive decision report.
    """
    
    def __init__(self, agent_coordinator=None):
        self.kg = GraphManager()
        self.agent_coordinator = agent_coordinator
        
    def generate_report(self, machine_id: str, virtual_rep: VirtualRepresentation) -> Dict[str, Any]:
        report = {
            "decision_id": str(uuid.uuid4()),
            "machine_id": machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # 1. Digital Twin Status
        health_score = TwinHealthMonitoring.calculate_health_score(virtual_rep)
        wear_factor = virtual_rep.machine.parameters.get("wear", 0.0)
        
        # 2. Risk Score & Failure Prediction
        # Risk score is inverse of health, amplified by wear
        risk_score = min(100.0, max(0.0, (100.0 - health_score) + (wear_factor * 20.0)))
        report["risk_score"] = round(risk_score, 2)
        
        if risk_score > 75.0:
            report["failure_prediction"] = "High Risk: Failure probable within next 100 simulation cycles."
        elif risk_score > 40.0:
            report["failure_prediction"] = "Medium Risk: Degradation accelerating."
        else:
            report["failure_prediction"] = "Low Risk: Operating normally."

        # 3. Reinforcement Learning Action
        rl_action = None
        # Naive default model lookup
        model_path = os.path.join(MODEL_DIR, "twin_agent_PPO.zip")
        if os.path.exists(model_path):
            try:
                rl_action = RLManager.get_action(virtual_rep, "PPO", model_path)
            except Exception:
                pass
                
        # 4. Knowledge Graph Reasoning
        # If risk is high, see if we have precedents
        kg_context = "No historical precedents required for current risk level."
        if risk_score > 40.0:
            past_issues = self.kg.reason_similar_failures("degradation")
            if past_issues:
                kg_context = f"Found {len(past_issues)} past issues. Most successful plan: {past_issues[0].get('past_successful_plan')}"
            else:
                kg_context = "Queried historical graphs, but no similar degradation paths found."

        # 5. Agentic AI Integration
        agent_plan = None
        if self.agent_coordinator and risk_score > 50.0:
            # Check if coordinator has a running plan
            if self.agent_coordinator.final_result:
                agent_plan = self.agent_coordinator.final_result.get("executed_plan")

        # 6. Synthesize Fused Output
        report["confidence_score"] = 95.0 if agent_plan else (85.0 if rl_action else 50.0)
        
        actions = []
        if rl_action:
            actions.append({"source": "RL Model", "action": rl_action, "rank": 1})
        if agent_plan:
            actions.append({"source": "Agent Swarm", "action": agent_plan, "rank": 2})
            
        report["alternative_actions"] = actions
        report["maintenance_recommendation"] = agent_plan if agent_plan else ("Schedule manual inspection." if risk_score > 75.0 else "None")
        
        # Explainability
        explanation = f"Decision driven by Health Score of {health_score}."
        if rl_action:
            explanation += f" AI Model suggests adjusting parameters by {rl_action}."
        explanation += f" {kg_context}"
        report["explainability"] = explanation

        return report
