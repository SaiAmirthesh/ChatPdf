import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from utils.config import Config

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def load_and_split_pdf(self, pdf_file) -> List[Document]:
        """Load PDF from uploaded file and split into chunks"""
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            chunks = self.text_splitter.split_documents(pages)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)