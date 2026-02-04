import os
import fitz  # PyMuPDF

from typing import List
from src.models import SourceDocument

def load_documents(directory_path: str) -> List[SourceDocument]:
    """
    Loads all PDF documents from the specified directory using PyMuPDF.
    Returns a list of SourceDocument objects (Pydantic).
    """
    documents = []
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                doc = fitz.open(file_path)
                file_docs = []
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        "file_path": file_path
                    }
                    file_docs.append(SourceDocument(page_content=text, metadata=metadata))
                
                documents.extend(file_docs)
                print(f"Loaded {len(file_docs)} pages from {filename}")
                doc.close()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents
