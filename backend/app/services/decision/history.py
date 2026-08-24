from typing import Dict, Any, List
from app.services.kg.graph_manager import GraphManager

class DecisionHistoryManager:
    """Manages tracking and replaying decisions via the Knowledge Graph."""
    
    def __init__(self):
        self.kg = GraphManager()

    def persist_decision(self, report: Dict[str, Any]) -> None:
        """
        Saves the decision report into the Neo4j Knowledge Graph as an AgentDecision.
        """
        # For simplicity, if there is a high risk, we record it as a failure resolution attempt
        machine_id = report["machine_id"]
        decision_id = report["decision_id"]
        risk_score = report["risk_score"]
        
        # Ensure asset exists
        self.kg.record_asset(machine_id, f"Asset-{machine_id}")
        
        if risk_score > 50.0:
            failure_id = f"evt_{decision_id}"
            self.kg.record_failure(machine_id, failure_id, report["explainability"])
            
            # Record decision
            plan = report["maintenance_recommendation"]
            if isinstance(plan, list):
                plan = str(plan)
            self.kg.record_agent_decision(failure_id, decision_id, plan)

    def get_decision_history(self, machine_id: str) -> List[Dict]:
        """
        Fetches the chronological decision history from the Graph.
        """
        return self.kg.get_asset_history(machine_id)
