import sys
import os
import warnings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress the Google Generative AI deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

from src.rag_engine import HealthcareRAG
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def test_single_generation():
    print("--- Testing Gemini Generation Validity ---")
    
    db_path = "./data/chroma_db_normal"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    print("Initializing RAG Engine...")
    rag = HealthcareRAG(persist_directory=db_path, model_name="gemini-2.5-flash-lite")
    
    test_question = "What is the primary basis of the recommendations in this guide?"
    print(f"\nTest Question: {test_question}")
    
    print("Generating answer...")
    try:
        response = rag.get_answer(test_question)
        
        print("\n--- RESULTS ---")
        print(f"Result Key Present: {'result' in response}")
        print(f"Source Docs Present: {len(response.get('source_documents', [])) > 0}")
        print("\n--- FULL GENERATED ANSWER ---")
        print(response.get("result", "NO RESULT FOUND"))
        
        print("\n--- CITATIONS (Source Documents) ---")
        for i, doc in enumerate(response.get("source_documents", [])):
            # Handle both dict and object access just in case
            if isinstance(doc, dict):
                meta = doc.get("metadata", {})
            else:
                meta = doc.metadata
                
            src = meta.get('source', 'unknown')
            page = meta.get('page', '?')
            print(f"{i+1}. {src} (Page {page})")

    except Exception as e:
        print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    test_single_generation()