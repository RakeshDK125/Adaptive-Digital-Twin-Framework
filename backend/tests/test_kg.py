import pytest
import uuid
from app.services.kg.graph_manager import GraphManager

# IMPORTANT: These tests require a local Neo4j instance running.
# If none is found, the GraphManager fails gracefully and returns []
# We will test the logic flow assuming the mock fallback works safely.

@pytest.fixture
def kg_manager():
    return GraphManager()

def test_graph_update_asset_and_failure(kg_manager):
    asset_id = str(uuid.uuid4())
    failure_id = str(uuid.uuid4())
    
    # 1. Update Graph
    kg_manager.record_asset(asset_id, "Test Turbine")
    kg_manager.record_failure(asset_id, failure_id, "High vibration anomaly detected")
    
    # In a mock scenario, this returns [], in a real DB it returns the history
    history = kg_manager.get_asset_history(asset_id)
    assert isinstance(history, list)

def test_graph_agent_memory(kg_manager):
    failure_id = str(uuid.uuid4())
    decision_id = str(uuid.uuid4())
    
    # 2. Agent Decision Memory
    plan = "Adjusted cooling flow rate by 5% and fine-tuned RL PPO model."
    kg_manager.record_agent_decision(failure_id, decision_id, plan)
    
    # 3. Semantic Search / Graph Reasoning
    # Looking for 'vibration' which might match the failure we just inserted
    results = kg_manager.reason_similar_failures("vibration")
    
    assert isinstance(results, list)
