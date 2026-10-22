from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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
            
    
    