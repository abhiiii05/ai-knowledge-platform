from fastapi import FastAPI, HTTPException,UploadFile, File, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.db import engine
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Document
import os
import shutil
from app.services import extract_text_from_pdf,chunk_text,get_embedding_model, store_embeddings_in_pinecone

UPLOAD_DIR = "uploads"
app = FastAPI()

os.makedirs(UPLOAD_DIR, exist_ok=True)



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

@app.post("/api/documents/")
def upload_documents(file:UploadFile = File(...), db: Session=Depends(get_db)):
    new_doc = Document(filename =  file.filename)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    base_name,extension = os.path.splitext(file.filename)
    safe_filename = str(new_doc.id)+extension
    filepath = os.path.join(UPLOAD_DIR,safe_filename)

    with open(filepath,"wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    new_doc.status = "processing"
    db.commit()
    
    extracted_text = extract_text_from_pdf(filepath)
    print(f"Extracted {len(extracted_text)} characters.")

    chunks = chunk_text(extracted_text)
    print(f"Split document into {len(chunks)} chunks.")

    # embedder = get_embedding_model()
    # sample_vector = embedder.embed_query(chunks[0])
    # print(f"Successfully generated a vector of length {len(sample_vector)}.")

    store_embeddings_in_pinecone(chunks, new_doc.id, new_doc.filename)
    print("Successfully uploaded vectors to Pinecone.")
        
    new_doc.status = "completed"
    db.commit()
    
    return {
            "id": new_doc.id,
            "filename": new_doc.filename,
            "status": new_doc.status
        }