import pytest
from app.agents.core.base import SharedMemory, MessageBus
from app.agents.roles.agents import (
    MonitoringAgent, ReasoningAgent, KnowledgeAgent, PlanningAgent,
    LearningAgent, OptimizationAgent, DecisionAgent, MemoryAgent, CoordinatorAgent
)

@pytest.fixture
def agent_swarm():
    memory = SharedMemory()
    bus = MessageBus()
    
    agents = {
        "monitoring": MonitoringAgent(memory, bus),
        "reasoning": ReasoningAgent(memory, bus),
        "knowledge": KnowledgeAgent(memory, bus),
        "planning": PlanningAgent(memory, bus),
        "learning": LearningAgent(memory, bus),
        "optimization": OptimizationAgent(memory, bus),
        "decision": DecisionAgent(memory, bus),
        "memory": MemoryAgent(memory, bus),
        "coordinator": CoordinatorAgent(memory, bus),
    }
    return memory, bus, agents

def test_agent_all_clear_flow(agent_swarm):
    memory, bus, agents = agent_swarm
    
    # Healthy telemetry
    memory.write("latest_telemetry", {"temp": 50.0, "vib": 2.0}, "Test")
    
    bus.publish("CoordinatorAgent", {"action": "EVALUATE_HEALTH"}, "Test")
    
    # Coordinator -> Monitoring
    assert len(bus.history) == 2
    assert bus.history[1]["topic"] == "MonitoringAgent"
    
    # Monitoring -> Coordinator (ALL_CLEAR)
    assert len(bus.history) == 3
    assert bus.history[2]["topic"] == "CoordinatorAgent"
    assert agents["coordinator"].final_result["message"] == "No anomalies detected."

def test_agent_anomaly_diagnosis_plan_flow(agent_swarm):
    memory, bus, agents = agent_swarm
    
    # Unhealthy telemetry
    memory.write("latest_telemetry", {"temp": 95.0, "vib": 6.0}, "Test")
    
    bus.publish("CoordinatorAgent", {"action": "EVALUATE_HEALTH"}, "Test")
    
    # Check that Memory recorded the anomaly and diagnosis
    assert memory.read("anomaly_detected") is True
    assert "High wear detected" in memory.read("current_diagnosis")
    
    # Check that a plan was proposed and approved
    assert memory.read("proposed_plan") is not None
    assert memory.read("approved_plan") is not None
    
    # Check final result
    assert agents["coordinator"].final_result["status"] == "success"
    assert len(agents["coordinator"].final_result["executed_plan"]) == 3
