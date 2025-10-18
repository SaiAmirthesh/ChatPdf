# ChatPDF - Q&A Assistant

A Streamlit-based application that allows you to upload any PDF document and ask questions about its content using AI.

## Features

- 📄 Upload any PDF document
- 🤖 AI-powered question answering
- 💬 Interactive chat interface
- 🔍 Semantic search using RAG architecture
- 🎯 Accurate, context-based responses

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Set Google API key in `.env` file
6. Run: `streamlit run app.py`

## Usage

1. Upload a PDF file using the sidebar
2. Click "Process PDF"
3. Start asking questions in the chat interface