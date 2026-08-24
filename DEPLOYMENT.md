# Adaptive Digital Twin - Deployment Guide

This document outlines how to deploy the entire ecosystem (Backend, Frontend, Databases, Message Brokers) using Docker Compose.

## Prerequisites
- Docker Engine & Docker Compose
- Git
- Node.js (for local UI dev only)

## Architecture Overview
The `docker-compose.yml` orchestrates 6 containers:
1. **`twin_backend`**: FastAPI / Python 3.12 (Port 8000)
2. **`twin_frontend`**: React / Nginx (Port 80)
3. **`twin_postgres`**: Telemetry DB (Port 5432)
4. **`twin_neo4j`**: Knowledge Graph (Port 7687/7474)
5. **`twin_redis`**: Message Broker (Port 6379)
6. **`twin_rabbitmq`**: Ingestion Queue (Port 5672)

## 1. Environment Setup
1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and change `SECRET_KEY` to a secure random string. This is used for JWT Authentication.

## 2. Launch the Stack
Run the entire production stack in detached mode:
```bash
docker compose up -d --build
```

## 3. Verify Deployment
- **React Dashboard**: Open `http://localhost`
- **FastAPI Swagger Docs**: Open `http://localhost:8000/docs`
- **Neo4j Browser**: Open `http://localhost:7474`
- **RabbitMQ Admin**: Open `http://localhost:15672`

## 4. Authentication (JWT)
Most API endpoints (RL, Agents, Decisions) are protected by JWT Auth.
1. Go to `http://localhost:8000/docs`
2. Click **Authorize**
3. Login using the default admin credentials:
   - **Username**: `admin`
   - **Password**: `admin123`
4. The Swagger UI will now automatically inject the Bearer token into all requests.

## 5. Load Testing
To verify the system can handle high-throughput IoT ingestion:
```bash
cd backend
locust -f tests/load_test.py --host=http://localhost:8000
```
Open `http://localhost:8089` to start the swarm.
