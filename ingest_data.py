import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_engine import HealthcareRAG
from src.utils import load_documents
from src.custom_loader import load_documents_custom
from dotenv import load_dotenv

def setup_databases():
    load_dotenv()
    
    docs_dir = "./data/documents/raw"
    
    # ---------------------------
    # 1. Normal Chunking Approach
    # ---------------------------
    print("\n--- Starting Normal Chunking Ingestion ---")
    normal_db_path = "./data/chroma_db_normal"
    
    # Clean up existing if needed (optional, here we just append/update or clean?)
    # For a clean slate comparison, let's remove if exists
    if os.path.exists(normal_db_path):
        print(f"Removing existing {normal_db_path}...")
        try:
             # Chroma requires closing connections or just forceful deletion if not running
             shutil.rmtree(normal_db_path) 
        except Exception as e:
            print(f"Warning: Could not remove {normal_db_path}: {e}")

    rag_normal = HealthcareRAG(persist_directory=normal_db_path)
    
    print("Loading documents (Standard)...")
    normal_docs = load_documents(docs_dir)
    
    if normal_docs:
        print(f"Ingesting {len(normal_docs)} pages into Normal DB...")
        rag_normal.ingest_documents(normal_docs)
        print("Normal DB ready.")
    else:
        print("No documents found for Normal DB.")

    # ---------------------------
    # 2. Customized Approach
    # ---------------------------
    print("\n--- Starting Customized Chunking Ingestion ---")
    custom_db_path = "./data/chroma_db_custom"
    images_dir = "./data/extracted_images"
    
    if os.path.exists(custom_db_path):
        print(f"Removing existing {custom_db_path}...")
        try:
            shutil.rmtree(custom_db_path)
        except Exception as e:
            print(f"Warning: Could not remove {custom_db_path}: {e}")
            
    # Clean images dir
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    
    rag_custom = HealthcareRAG(persist_directory=custom_db_path)
    
    print("Loading documents (Custom w/ Headlines & Images)...")
    custom_docs = load_documents_custom(docs_dir, images_dir)
    
    if custom_docs:
        print(f"Ingesting {len(custom_docs)} pages into Custom DB...")
        # We reuse the same ingestion logic (splitting) but now our input docs satisfy the 
        # "store headlines, page numbers, images" requirement via metadata.
        rag_custom.ingest_documents(custom_docs)
        print("Custom DB ready.")
    else:
        print("No documents found for Custom DB.")

if __name__ == "__main__":
    setup_databases()
