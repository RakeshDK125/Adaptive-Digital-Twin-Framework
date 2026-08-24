import os
from neo4j import GraphDatabase

class Neo4jClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            try:
                cls._instance.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j. Tests should use mock. {e}")
                cls._instance.driver = None
        return cls._instance

    def close(self):
        if self.driver:
            self.driver.close()

    def get_session(self):
        if self.driver:
            return self.driver.session()
        raise ConnectionError("Neo4j Driver not initialized.")
