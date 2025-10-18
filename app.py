import streamlit as st
import os
from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStoreManager
from core.rag_agent import RAGAgent
from langchain_core.tools import tool
import tempfile
from utils.config import Config

st.set_page_config(
    page_title="ChatPDF",
    layout="wide"
)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def initialize_rag_system(documents):
    """Initialize the RAG system with processed documents"""
    try:
        vector_store_manager = VectorStoreManager()
        vectorstore = vector_store_manager.create_vector_store(
            documents, 
            collection_suffix=st.session_state.get('file_hash', 'default')
        )
        
        retriever = vector_store_manager.get_retriever(vectorstore, k=5)
        
        @tool
        def retriever_tool(query: str) -> str:
            """Search and return information from the uploaded PDF document."""
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the document."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document excerpt {i+1}:\n{doc.page_content}")
            
            return "\n\n".join(results)
        
        rag_agent = RAGAgent(retriever_tool)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.rag_agent = rag_agent
        st.session_state.pdf_processed = True
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

def main():
    st.title("ChatPDF - Your Q&A Assistant")
    st.markdown("Upload any PDF document and ask questions about its content!")
    
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF file to ask questions about its content"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor()
                        documents = processor.load_and_split_pdf(uploaded_file)
                        
                        st.success(f"PDF processed! Created {len(documents)} document chunks.")
                        
                        # Initialize RAG system
                        if initialize_rag_system(documents):
                            st.success("RAG system initialized successfully!")
                            st.session_state.chat_history = []  # Clear previous chat
                            
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        st.markdown("---")
        st.header("Instructions")
        st.markdown("""
        1. Upload a PDF file
        2. Click 'Process PDF'
        3. Start asking questions in the chat
        4. The AI will answer based on the document content
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat with your PDF")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask a question about your PDF..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if not st.session_state.pdf_processed:
                with st.chat_message("assistant"):
                    st.error("Please upload and process a PDF file first!")
                return
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_agent.query(prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.header("Status")
        
        if st.session_state.pdf_processed:
            st.success("PDF Processed")
            st.info("Ready for questions!")
        else:
            st.warning("Waiting for PDF upload")
        
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("Reset System"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    if not Config.GOOGLE_API_KEY:
        st.error("Please set GOOGLE_API_KEY in your .env file")
    else:
        main()