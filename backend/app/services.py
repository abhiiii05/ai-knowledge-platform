import re
from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from functools import lru_cache

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
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 200,
    #     length_function = len
    # 
    # )
    embedder = get_embedding_model()
    splitter = SemanticChunker(embedder)
    return splitter.split_text(text)

@lru_cache(maxsize=1)
def get_embedding_model():
    print("INITIALIZING: Connecting to Google Cloud Embeddings (3072-dim)...")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
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

def generate_answer(query : str, retrieved_chunks: list[dict])-> str:
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks])

    llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=os.getenv("GROQ_API_KEY")
        )

    prompt = PromptTemplate(
            template="""You are a helpful technical assistant. Use ONLY the provided context to answer the user's question.
            If the answer is not in the context, say "I cannot answer this based on the provided documents."

            Context:
            {context}

            Question:
            {question}

            Answer:""",
            input_variables=["context", "question"]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})

    if isinstance(response.content, list):
            return response.content[0].get("text", "")
    return response.content

    return response.content
