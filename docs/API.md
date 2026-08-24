# API Reference

The Adaptive Digital Twin exposes a REST API via FastAPI and a real-time WebSocket connection.

## Base URL
`http://localhost:8000/api/v1`

## Security
All Endpoints under `/rl`, `/agents`, and `/decisions` require JWT Authentication.
1. POST `/auth/token`: Provide `username: admin`, `password: admin123` to receive a Bearer token.
2. Inject `Authorization: Bearer <token>` in the header of subsequent requests.

## Endpoints

### Digital Twin
- `POST /twins/ingest`: Ingests a new telemetry payload (MQTT simulation).
- `GET /twins/{machine_id}`: Retrieves the synchronized Virtual Representation state.

### Reinforcement Learning (Protected)
- `POST /rl/{machine_id}/train`: Triggers the training sequence (PPO/SAC) based on current physics.
- `GET /rl/{machine_id}/action`: Returns the mathematically optimal continuous control action.

### Agentic Swarm (Protected)
- `POST /agents/diagnose`: Triggers the Multi-Agent Swarm (Reasoning, Knowledge, Planning agents).
- `GET /agents/memory`: Dumps the current Shared Blackboard Context.

### Explainable AI (XAI)
- `GET /xai/{machine_id}/feature-importance`: Returns exact SHAP values dictating the RL action.
- `POST /xai/counterfactual`: Accepts a "What-If" parameter JSON and returns the Delta action.

## WebSockets
Connect to `ws://localhost:8000/ws/{machine_id}` to receive real-time streaming telemetry and Decision Engine updates.
