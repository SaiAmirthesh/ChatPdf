import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List
import shutil
from utils.config import Config

class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.persist_directory = Config.PERSIST_DIRECTORY
        self.collection_name = Config.COLLECTION_NAME
        
    def create_vector_store(self, documents: List[Document], collection_suffix: str = ""):
        """Create a new vector store from documents"""
        
        full_collection_name = f"{Config.COLLECTION_NAME}_{collection_suffix}" if collection_suffix else Config.COLLECTION_NAME
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=full_collection_name
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def get_retriever(self, vectorstore, k: int = 5):
        """Get retriever from vector store"""
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def cleanup(self):
        """Clean up vector store directory"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)