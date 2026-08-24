import uuid
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timezone

class SharedMemory:
    """A centralized blackboard for agents to share context."""
    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []

    def write(self, key: str, value: Any, agent_name: str) -> None:
        self.context[key] = value
        self.logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "action": "WRITE",
            "key": key,
            "value": value
        })

    def read(self, key: str) -> Any:
        return self.context.get(key)
        
    def get_all(self) -> Dict[str, Any]:
        return self.context
        
    def clear(self) -> None:
        self.context = {}

class MessageBus:
    """Simple pub/sub router for inter-agent communication."""
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.history: List[Dict[str, Any]] = []

    def subscribe(self, topic: str, callback: Callable) -> None:
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, message: Dict[str, Any], sender: str) -> None:
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender": sender,
            "topic": topic,
            "message": message
        })
        for callback in self.subscribers.get(topic, []):
            # In a real async framework, this would be scheduled on an event loop
            callback(message)

class BaseAgent:
    """Abstract base class for all specialized agents."""
    def __init__(self, name: str, memory: SharedMemory, bus: MessageBus):
        self.name = name
        self.memory = memory
        self.bus = bus
        self.tools: Dict[str, Callable] = {}
        self.bus.subscribe(self.name, self.process_message)

    def register_tool(self, tool_name: str, func: Callable) -> None:
        self.tools[tool_name] = func

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        raise ValueError(f"Tool {tool_name} not found in agent {self.name}")

    def reflect(self, task: str, outcome: Any) -> bool:
        """Self-correction hook. Agents can override this to check if a task succeeded."""
        return True

    def process_message(self, message: Dict[str, Any]) -> None:
        """Must be implemented by subclasses to handle incoming tasks."""
        raise NotImplementedError
