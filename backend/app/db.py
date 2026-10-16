from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import os 
from sqlalchemy.orm import sessionmaker, declarative_base

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    echo=False,  
    pool_pre_ping=True  
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()