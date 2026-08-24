import asyncio
import json
import pandas as pd
from typing import Dict, Any, Callable, Coroutine
from app.services.ingestion.pipeline import PreprocessingPipeline
from app.models.telemetry import TelemetryRecord
from app.db.session import SessionLocal

# Mock imports for the connectors to avoid crashing if actual brokers aren't running
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    from aiokafka import AIOKafkaConsumer
except ImportError:
    AIOKafkaConsumer = None

class IngestionManager:
    """Orchestrates ingestion, preprocessing, and DB storage."""
    def __init__(self, twin_sync_callback: Callable[[str, Dict[str, float]], None]):
        self.pipeline = PreprocessingPipeline()
        self.twin_sync_callback = twin_sync_callback

    def handle_payload(self, raw_payload: Dict[str, Any]):
        """Processes payload, saves to DB, and syncs with twin."""
        try:
            # 1. Process
            result = self.pipeline.process(raw_payload)
            machine_id = result["machine_id"]
            timestamp = result["timestamp"]
            
            # 2. Extract values for the Twin Engine
            twin_payload = {}
            db_records = []
            
            for sensor, data in result["sensors"].items():
                twin_payload[sensor] = data["processed_value"]
                
                # Create DB Record
                record = TelemetryRecord(
                    machine_id=machine_id,
                    timestamp=timestamp,
                    sensor_name=sensor,
                    raw_value=data["raw_value"],
                    processed_value=data["processed_value"],
                    is_outlier=data["is_outlier"]
                )
                db_records.append(record)

            # 3. Store to DB
            db = SessionLocal()
            try:
                db.bulk_save_objects(db_records)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"DB Error: {e}")
            finally:
                db.close()

            # 4. Sync with Digital Twin
            self.twin_sync_callback(machine_id, twin_payload)
            
        except Exception as e:
            print(f"Ingestion Error: {e}")


# --- Connectors ---

class CSVConnector:
    """Batch loads historical data from CSV into the pipeline."""
    def __init__(self, manager: IngestionManager):
        self.manager = manager
        
    def ingest_file(self, file_path: str, machine_id: str):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            payload = {
                "machine_id": machine_id,
                "sensors": row.to_dict() # Assuming all columns are sensors
            }
            if 'timestamp' in payload['sensors']:
                payload['timestamp'] = payload['sensors'].pop('timestamp')
            
            self.manager.handle_payload(payload)

class MQTTConnector:
    """Subscribes to MQTT topics."""
    def __init__(self, broker: str, port: int, topic: str, manager: IngestionManager):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.manager = manager
        if mqtt:
            self.client = mqtt.Client()
            self.client.on_message = self.on_message
            
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.manager.handle_payload(payload)
        except Exception:
            pass

    def start(self):
        if mqtt:
            self.client.connect(self.broker, self.port, 60)
            self.client.subscribe(self.topic)
            self.client.loop_start()

class KafkaConnector:
    """Async Kafka Consumer."""
    def __init__(self, bootstrap_servers: str, topic: str, manager: IngestionManager):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.manager = manager

    async def start(self):
        if not AIOKafkaConsumer:
            return
            
        consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        await consumer.start()
        try:
            async for msg in consumer:
                self.manager.handle_payload(msg.value)
        finally:
            await consumer.stop()
