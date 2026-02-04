import json
import time
import os
import sys
import pandas as pd
# from langchain_community.callbacks import get_openai_callback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG

def load_test_set(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate():
    print("Initializing RAG Engine for Evaluation...")
    rag = HealthcareRAG()
    
    test_set = load_test_set("evaluation/test_set.json")
    results = []
    
    total_cost = 0.0
    total_tokens = 0
    
    print(f"Starting evaluation on {len(test_set)} questions...")
    
    for i, item in enumerate(test_set):
        question = item['question']
        expected = item['expected_answer']
        
        start_time = time.time()
        
        # with get_openai_callback() as cb:
        try:
            response = rag.get_answer(question)
            answer = response['result']
            # Basic cost tracking placeholder (Gemini free tier or manual calc needed)
            cost = 0.0 
            tokens = 0 
        except Exception as e:
            answer = f"Error: {str(e)}"
            cost = 0
            tokens = 0
        
        end_time = time.time()
        latency = end_time - start_time
        
        total_cost += cost
        total_tokens += tokens
        
        print(f"[{i+1}/{len(test_set)}] Latency: {latency:.2f}s | Cost: ${cost:.5f}")
        
        results.append({
            "question": question,
            "answer": answer,
            "expected": expected,
            "latency_seconds": latency,
            "cost_usd": cost,
            "tokens": tokens,
            "source_docs": [os.path.basename(d.metadata.get('source', 'unknown')) for d in response.get('source_documents', [])]
        })

    # Summary Statistics
    df = pd.DataFrame(results)
    avg_latency = df['latency_seconds'].mean()
    p95_latency = df['latency_seconds'].quantile(0.95)
    total_cost_1k = (total_cost / len(test_set)) * 1000
    
    print("\n=== Evaluation Results ===")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"P95 Latency: {p95_latency:.2f}s")
    print(f"Total Cost for this run: ${total_cost:.5f}")
    print(f"Estimated Cost per 1,000 questions: ${total_cost_1k:.4f}")
    
    # Save Report
    report = {
        "metrics": {
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "total_cost": total_cost,
            "est_cost_per_1k": total_cost_1k
        },
        "details": results
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_report.json and evaluation_results.csv")

if __name__ == "__main__":
    evaluate()
