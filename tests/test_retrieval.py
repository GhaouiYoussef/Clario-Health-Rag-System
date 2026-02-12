import sys
import os
import warnings
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from src.rag_engine import HealthcareRAG
from dotenv import load_dotenv

def test_retrieval_performance():
    # Load env variables
    load_dotenv()
    
    print("=== Testing Advanced Hybrid Retrieval & Reranking ===")
    
    # Use the custom database if it exists, otherwise fallback to normal
    db_path = "./data/chroma_db_chap_based"
    if not os.path.exists(db_path):
        db_path = "./data/chroma_db_normal"
        print(f"⚠️ {db_path} (VI) not found, using {db_path} instead.")
    else:
        print(f"✅ Using Database: {db_path}")

    print("Initializing RAG Engine (this might take a moment to build BM25 index)...")
    try:
        rag = HealthcareRAG(persist_directory=db_path)
    except Exception as e:
        print(f"❌ Failed to initialize RAG engine: {e}")
        return

    # Configuration matches user request: 40 initially retrieved, 20 after reranking
    k_initial = 40
    n_final = 20
    doc_div = 0.2 # Max 50% results from one doc
    
    test_queries = [
        # "What are the steps for CPR in an infant?",
        # "Symptoms of a heart attack and recommended actions",
        # "How to use an AED on a child?",
        # "What are the primary components of a 'Head-to-Toe' check for a conscious person?"
        "how shgould i cleani my needle after giving insulin injection"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- Test Case {i+1}: '{query}' ---")
        print(f"Retrieving {k_initial} candidates then reranking to top {n_final} (Doc Diversity: {doc_div})...")
        
        try:
            results = rag.hybrid_retrieval(
                query=query, 
                n_results=n_final, 
                k_candidates=k_initial,
                doc_diversity_ratio=doc_div
            )
            
            if not results:
                print("❌ No results found.")
                continue

            print(f"Found {len(results)} results after reranking and diversity filtering:")
            
            for rank, hit in enumerate(results):
                content = hit.get('content', '')[:100].replace('\n', ' ') + "..."
                meta = hit.get('metadata', {})
                score = hit.get('final_score', 0.0)
                title = hit.get('title_txt', 'No Title')
                source = meta.get('source', 'Unknown')
                page = meta.get('page_range', meta.get('page', '?'))
                document_name = meta.get('doc_name', 'Unknown Document')
                
                print(f"  [{rank+1}] Score: {score:.4f} | {title} | {source} (P.{page}) | Document: {document_name}")
                print(f"      Text sample: {content}")
                
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")

if __name__ == "__main__":
    test_retrieval_performance()
