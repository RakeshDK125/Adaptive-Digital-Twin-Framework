# Architecture & Theoretical Foundation

This document outlines the core theoretical pillars of the Adaptive Digital Twin Framework.

## 1. Meta-Reinforcement Learning (Meta-RL)
Traditional Deep Reinforcement Learning (e.g., PPO, SAC) requires thousands of episodes to converge on an optimal control policy. In an industrial setting, machine physics dynamically change due to wear and tear (concept drift). 
We wrap our SB3 models in a **Model-Agnostic Meta-Learning (MAML)** abstraction (`backend/app/research/meta_rl.py`). This forces the agent to learn a set of initialization weights that can rapidly adapt (few-shot learning) to a new fault state with minimal gradient steps.

## 2. Multi-Agent Swarm (Agentic AI)
Continuous control (RL) cannot solve discrete reasoning tasks (e.g., "Schedule a maintenance worker, order part X, and update the graph").
We utilize a 9-Agent Swarm (`backend/app/agents/roles/agents.py`) coordinated by a Central Message Bus. 
- **Monitoring Agent**: Listens to the telemetry pipeline.
- **Reasoning Agent**: Diagnoses root causes when the `ADWINDriftDetector` fires.
- **Planning Agent**: Constructs sequential physical repair plans.

## 3. Topological Knowledge Graph
All decisions are permanently recorded in Neo4j. The `TopologicalReasoner` allows the swarm to query:
`FAULT -> affects -> COMPONENT -> historically_fixed_by -> AGENT_DECISION`
This provides the swarm with long-term memory, enabling it to bypass RL exploration if a historically highly-confident mitigation policy exists.

## 4. Federated Learning (Edge Sync)
The architecture supports multi-site deployments. The `FederatedLearningAggregator` simulates FedAvg, allowing multiple identical edge twins to securely aggregate their Meta-RL policy weights without sharing raw proprietary telemetry data.
