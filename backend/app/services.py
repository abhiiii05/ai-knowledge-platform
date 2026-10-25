import re

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

index_name = "knowledge-platform"

def extract_text_from_pdf(filepath: str) -> str:
    full_text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text+=text+"\n"
    return full_text

def chunk_text(text:str)-> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    return splitter.split_text(text)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2")

def store_embeddings_in_pinecone(chunks:list[str],document_id:str, filename:str):
    embedder =get_embedding_model()
    index_name = "knowledge-platform"

    metadatas = [{"document_id":str(document_id), "filename":filename} for _ in chunks]

    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embedder,
        index_name=index_name,
        metadatas=metadatas
    )

def search_documents(query : str, top_k: int = 3) -> list[dict]:
    embedder = get_embedding_model()
    index_name = "knowledge-platform"

    vectorstore = PineconeVectorStore(
        index_name = index_name,
        embedding = embedder
    )

    results = vectorstore.similarity_search(query,k=top_k)

    formatted_results = []
    for doc in results:
        formatted_results.append({
            "content" : doc.page_content,
            "source" : doc.metadata.get("filename", "Unknown")
        })
    return formatted_results
    