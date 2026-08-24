# Installation Guide

## Production Setup (Docker)
The entire Adaptive Digital Twin architecture runs on Docker.
1. Clone the repository.
2. Ensure Docker Desktop is running.
3. Setup environments:
```bash
cp .env.example .env
```
4. Build and boot the 6-container stack:
```bash
docker compose up -d --build
```
This will automatically launch FastAPI, Vite/React, Postgres, Neo4j, Redis, and RabbitMQ.

## Bare-Metal Setup (Development)

### Backend
Requires Python 3.12.
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
Requires Node.js 20+.
```bash
cd frontend
npm install
npm run dev
```

### Databases
If not using Docker, you must independently install and configure PostgreSQL on port 5432 and Neo4j on port 7687, and update your `.env` file accordingly.
