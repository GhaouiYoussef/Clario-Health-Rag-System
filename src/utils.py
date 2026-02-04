import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def load_documents(directory_path: str) -> List[Document]:
    """Loads all PDF documents from the specified directory."""
    documents = []
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents
