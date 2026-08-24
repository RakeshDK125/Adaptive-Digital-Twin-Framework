import uuid
from sqlalchemy import Column, String, Float, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from datetime import datetime, timezone
from app.db.session import Base

class TelemetryRecord(Base):
    __tablename__ = "telemetry"

    # Use a string UUID for sqlite compatibility during tests
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    machine_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    sensor_name = Column(String, index=True, nullable=False)
    raw_value = Column(Float, nullable=False)
    processed_value = Column(Float, nullable=False)
    is_outlier = Column(Boolean, default=False)
