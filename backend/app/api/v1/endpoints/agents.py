from fastapi import APIRouter, BackgroundTasks
from typing import Dict, Any, List
import time

from app.agents.core.base import SharedMemory, MessageBus
from app.agents.roles.agents import (
    MonitoringAgent, ReasoningAgent, KnowledgeAgent, PlanningAgent,
    LearningAgent, OptimizationAgent, DecisionAgent, MemoryAgent, CoordinatorAgent
)

router = APIRouter()

# Global instances for the framework
memory = SharedMemory()
bus = MessageBus()

# Initialize the Swarm
monitoring = MonitoringAgent(memory, bus)
reasoning = ReasoningAgent(memory, bus)
knowledge = KnowledgeAgent(memory, bus)
planning = PlanningAgent(memory, bus)
learning = LearningAgent(memory, bus)
optimization = OptimizationAgent(memory, bus)
decision = DecisionAgent(memory, bus)
memory_agent = MemoryAgent(memory, bus)
coordinator = CoordinatorAgent(memory, bus)

@router.post("/task")
def submit_task(task_type: str = "EVALUATE_HEALTH", telemetry_mock: Dict[str, float] = None):
    """
    Submits a task to the Coordinator Agent.
    """
    if telemetry_mock:
        memory.write("latest_telemetry", telemetry_mock, "API")
        
    coordinator.final_result = None
    bus.publish("CoordinatorAgent", {"action": task_type}, "API")
    
    # Wait for coordinator to synthesize result (mock sync wait for demo purposes)
    timeout = 10
    while coordinator.final_result is None and timeout > 0:
        time.sleep(0.1)
        timeout -= 1
        
    if coordinator.final_result:
        return coordinator.final_result
    return {"status": "processing", "message": "Task delegated to swarm."}

@router.get("/memory")
def get_shared_memory():
    """Returns the current state of the Blackboard/Shared Memory."""
    return memory.get_all()

@router.get("/logs")
def get_message_logs():
    """Returns the communication transcript between the agents."""
    return {"message_bus_history": bus.history, "memory_write_logs": memory.logs}

@router.post("/cleanup")
def cleanup_memory():
    """Tells the Memory agent to clear the context."""
    bus.publish("MemoryAgent", {"action": "CLEANUP"}, "API")
    return {"message": "Memory cleared."}
