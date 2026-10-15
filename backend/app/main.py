from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.db import engine


app = FastAPI()


@app.get("/health")
def health():
    return {"status":"ok"}


@app.get("/db-health")
def db_health() :
    try : 
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        return {"database_status": "connected"}
    except SQLAlchemyError as e:
            print(f"Database connection failed: {e}")
            raise HTTPException(status_code=503, detail="Database connection failed")