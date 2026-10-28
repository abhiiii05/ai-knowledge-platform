import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Note: Updated to match your services.py index name
index_name = "knowledge-platform"

if index_name in pc.list_indexes().names():
    print(f"Deleting old index: {index_name}...")
    pc.delete_index(index_name)
    time.sleep(5)

# Note: Updated to 3072 to match gemini-embedding-2
print(f"Creating new 3072-dimension index: {index_name}...")
pc.create_index(
    name=index_name,
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("Success! The new index is ready.")
