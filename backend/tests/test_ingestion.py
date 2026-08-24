import pytest
import numpy as np
from app.services.ingestion.pipeline import PreprocessingPipeline
from app.services.ingestion.connectors import IngestionManager
from app.db.session import SessionLocal, Base, engine
from app.models.telemetry import TelemetryRecord

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_preprocessing_pipeline():
    pipeline = PreprocessingPipeline(window_size=5, outlier_z_threshold=2.0)
    machine_id = "test_machine"
    
    # 1. Normal data
    res1 = pipeline.process({"machine_id": machine_id, "sensors": {"temp": 50.0}})
    assert res1["sensors"]["temp"]["processed_value"] == 50.0
    assert res1["sensors"]["temp"]["is_outlier"] is False

    # Fill buffer to enable outlier detection (needs >= 3)
    pipeline.process({"machine_id": machine_id, "sensors": {"temp": 51.0}})
    pipeline.process({"machine_id": machine_id, "sensors": {"temp": 49.0}})

    # 2. Outlier Spike
    res_outlier = pipeline.process({"machine_id": machine_id, "sensors": {"temp": 150.0}})
    assert res_outlier["sensors"]["temp"]["is_outlier"] is True
    
    # 3. Missing Value Handling (Forward fill)
    res_missing = pipeline.process({"machine_id": machine_id, "sensors": {"temp": np.nan, "pressure": 10.0}})
    # temp should be filled with the EMA result of the outlier step
    assert res_missing["sensors"]["temp"]["processed_value"] == res_outlier["sensors"]["temp"]["processed_value"]

def test_ingestion_manager():
    synced_data = {}
    def mock_sync(machine_id, payload):
        synced_data[machine_id] = payload
        
    manager = IngestionManager(twin_sync_callback=mock_sync)
    
    payload = {
        "machine_id": "machine_x",
        "sensors": {"vibration": 2.5}
    }
    
    manager.handle_payload(payload)
    
    # Verify Callback
    assert "machine_x" in synced_data
    assert synced_data["machine_x"]["vibration"] == 2.5
    
    # Verify DB insertion
    db = SessionLocal()
    record = db.query(TelemetryRecord).filter(TelemetryRecord.machine_id == "machine_x").first()
    assert record is not None
    assert record.sensor_name == "vibration"
    assert record.raw_value == 2.5
    db.close()
