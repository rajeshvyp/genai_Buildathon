GenAI RAG Agriculture Scheme Chatbot

Project Overview:
This project is a Generative AI-based Retrieval-Augmented Generation (RAG) chatbot designed to help users get information about agricultural schemes in Tamil Nadu.

The chatbot retrieves relevant information from agriculture documents and uses an OpenAI language model to generate answers with source references.

Technologies Used
- Python
- Streamlit
- LangChain
- FAISS Vector Database
- OpenAI API
- PyMuPDF and Tesseract OCR

 Python Files
1. app.py – Provides the Streamlit user interface for interacting with the chatbot.
2. rag_chatbot.py – Implements document retrieval and AI-powered answer generation.
3. process_documents.py – Extracts text from PDF and TXT documents, using OCR when required.
4. build_vectorstore.py – Converts processed documents into embeddings and builds the FAISS vector index.

Data
The data/raw folder contains the source documents used as the agriculture knowledge base.

How to Run
1. Install the required Python packages.
2. Configure your OpenAI API key in a .env file in the project root.
3. Ensure the required FAISS vector index files are available.
4. Run the following command from the project root:

powershell
python -m streamlit run .\chatbot\app.py


5. Open the local URL shown in the terminal.

Purpose
The project demonstrates how RAG can help users access agriculture scheme information through a simple conversational interface.
