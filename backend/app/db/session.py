import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use SQLite as a fallback for local testing if Postgres is not available
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./digital_twin_test.db")

engine = create_engine(
    DATABASE_URL, 
    # check_same_thread=False is needed only for SQLite
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
