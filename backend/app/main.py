from fastapi import FastAPI, Depends
from app.api.v1.endpoints import twins, rl, agents, decisions, xai, auth
from app.db.session import engine
from app.models.telemetry import Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Adaptive Digital Twin API",
    description="API for Reinforcement Learning and Agentic AI Digital Twin",
    version="1.0.0",
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(twins.router, prefix="/api/v1/twins", tags=["twins"])
app.include_router(rl.router, prefix="/api/v1/rl", tags=["rl"], dependencies=[Depends(auth.get_current_user)])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"], dependencies=[Depends(auth.get_current_user)])
app.include_router(decisions.router, prefix="/api/v1/decisions", tags=["decisions"], dependencies=[Depends(auth.get_current_user)])
app.include_router(xai.router, prefix="/api/v1/xai", tags=["xai"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Adaptive Digital Twin API"}
