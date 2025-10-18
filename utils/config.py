import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "pdf_documents"