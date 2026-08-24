from typing import Dict, Any, List
from app.db.neo4j_client import Neo4jClient

class GraphManager:
    """Manages Graph Operations, Reasoning, and Semantic Memory."""
    
    def __init__(self):
        self.client = Neo4jClient()

    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        try:
            with self.client.get_session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except ConnectionError:
            # Fallback for mock environments
            return []

    # --- Graph Update Methods ---
    def record_asset(self, asset_id: str, name: str) -> None:
        query = """
        MERGE (a:Asset {id: $id})
        SET a.name = $name
        """
        self._execute_query(query, {"id": asset_id, "name": name})

    def record_failure(self, asset_id: str, failure_id: str, description: str) -> None:
        query = """
        MATCH (a:Asset {id: $asset_id})
        MERGE (f:Failure {id: $failure_id})
        SET f.description = $description
        MERGE (a)-[:EXPERIENCED]->(f)
        """
        self._execute_query(query, {"asset_id": asset_id, "failure_id": failure_id, "description": description})

    def record_agent_decision(self, failure_id: str, decision_id: str, plan_summary: str) -> None:
        query = """
        MATCH (f:Failure {id: $failure_id})
        MERGE (d:AgentDecision {id: $decision_id})
        SET d.plan = $plan_summary
        MERGE (f)-[:RESOLVED_BY]->(d)
        """
        self._execute_query(query, {"failure_id": failure_id, "decision_id": decision_id, "plan_summary": plan_summary})

    # --- Graph Query & Reasoning Methods ---
    def get_asset_history(self, asset_id: str) -> List[Dict]:
        """Historical Memory: Fetches all failures and decisions for an asset."""
        query = """
        MATCH (a:Asset {id: $asset_id})-[:EXPERIENCED]->(f:Failure)
        OPTIONAL MATCH (f)-[:RESOLVED_BY]->(d:AgentDecision)
        RETURN f.id AS failure_id, f.description AS description, d.plan AS resolution_plan
        """
        return self._execute_query(query, {"asset_id": asset_id})

    def reason_similar_failures(self, keyword: str) -> List[Dict]:
        """Semantic Search / Reasoning: Finds past resolutions based on text matching in Failure."""
        query = """
        MATCH (f:Failure)-[:RESOLVED_BY]->(d:AgentDecision)
        WHERE toLower(f.description) CONTAINS toLower($keyword)
        RETURN f.description AS failure, d.plan AS past_successful_plan
        """
        return self._execute_query(query, {"keyword": keyword})
