from typing import Dict, Any, List
from app.agents.core.base import BaseAgent, SharedMemory, MessageBus

class MonitoringAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("MonitoringAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        action = message.get("action")
        if action == "CHECK_TELEMETRY":
            # Mock behavior: Fetching telemetry
            telemetry = self.memory.read("latest_telemetry")
            if telemetry and any(v > 80.0 for v in telemetry.values()):
                self.memory.write("anomaly_detected", True, self.name)
                self.bus.publish("ReasoningAgent", {"action": "DIAGNOSE", "data": telemetry}, self.name)
            else:
                self.bus.publish("CoordinatorAgent", {"status": "ALL_CLEAR"}, self.name)

class ReasoningAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("ReasoningAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "DIAGNOSE":
            data = message.get("data")
            # Mock diagnosis logic
            diagnosis = f"High wear detected due to values: {data}"
            self.memory.write("current_diagnosis", diagnosis, self.name)
            self.bus.publish("PlanningAgent", {"action": "CREATE_PLAN", "diagnosis": diagnosis}, self.name)

from app.services.kg.graph_manager import GraphManager

class KnowledgeAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("KnowledgeAgent", memory, bus)
        self.kg = GraphManager()
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "QUERY_GRAPH":
            query = message.get("query", "")
            
            # Use Graph Reasoning / Semantic Search on past failures
            past_resolutions = self.kg.reason_similar_failures(query)
            
            context = "No historical precedents found."
            if past_resolutions:
                plan = past_resolutions[0].get("past_successful_plan")
                context = f"Historical memory indicates similar issue resolved by: {plan}"
                
            self.bus.publish(message["reply_to"], {"context": context}, self.name)

class PlanningAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("PlanningAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "CREATE_PLAN":
            diagnosis = message.get("diagnosis")
            plan = [
                {"step": 1, "task": "Query historical context"},
                {"step": 2, "task": "Trigger RL Online Finetuning"},
                {"step": 3, "task": "Adjust Operational Speed"}
            ]
            self.memory.write("proposed_plan", plan, self.name)
            self.bus.publish("DecisionAgent", {"action": "REVIEW_PLAN", "plan": plan}, self.name)

class LearningAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("LearningAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "FINETUNE_RL":
            self.memory.write("rl_status", "Finetuning completed", self.name)
            self.bus.publish("CoordinatorAgent", {"status": "FINETUNED"}, self.name)

class OptimizationAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("OptimizationAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        pass # Evaluates "what-if" scenarios

class DecisionAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("DecisionAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "REVIEW_PLAN":
            plan = message.get("plan")
            # Safety gate logic
            is_safe = True
            if is_safe:
                self.memory.write("approved_plan", plan, self.name)
                self.bus.publish("CoordinatorAgent", {"action": "EXECUTE_PLAN", "plan": plan}, self.name)
            else:
                self.bus.publish("PlanningAgent", {"action": "REPLAN", "reason": "Unsafe thresholds"}, self.name)

class MemoryAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("MemoryAgent", memory, bus)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "CLEANUP":
            self.memory.clear()

class CoordinatorAgent(BaseAgent):
    def __init__(self, memory: SharedMemory, bus: MessageBus):
        super().__init__("CoordinatorAgent", memory, bus)
        self.final_result = None
        
    def process_message(self, message: Dict[str, Any]) -> None:
        if message.get("action") == "EVALUATE_HEALTH":
            # Entry point from API
            self.final_result = None
            self.bus.publish("MonitoringAgent", {"action": "CHECK_TELEMETRY"}, self.name)
        elif message.get("action") == "EXECUTE_PLAN":
            plan = message.get("plan")
            self.final_result = {"status": "success", "executed_plan": plan}
        elif message.get("status") == "ALL_CLEAR":
            self.final_result = {"status": "success", "message": "No anomalies detected."}
