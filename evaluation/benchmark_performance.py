import json
import time
import os
import sys
# import pandas as pd
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG
with open('./evaluation/costs_gemini.json', 'r') as f:
    models=json.load(f).get("models", {})

model_name = 'gemini-2.5-flash-lite'
MODEL_params = models.get(model_name, {})

MODEL = model_name
COST_PER_1M_INPUT_TOKENS = MODEL_params.get("embedding_cost_per_1M_tokens", 0.1) 
COST_PER_1M_OUTPUT_TOKENS = MODEL_params.get("generation_cost_per_1M_tokens", 0.4) 


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
    return input_cost + output_cost

def load_qa_pairs(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        data = json.load(f)
        # Handle if the file is a list of dicts directly
        if isinstance(data, list):
            return data
        return []

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_performance():
    print("--- Performance Evaluation (Normal Chunking) ---")
    
    # Initialize RAG with Normal DB
    db_path = "./data/chroma_db_normal"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} does not exist. Please run ingest_data.py first.")
        return

    print(f"Loading RAG Engine with DB: {db_path}")
    rag = HealthcareRAG(persist_directory=db_path, model_name=MODEL)
    
    # Load Questions
    qa_file = "./data/QA/Outpatient-GPT.json"
    print(f"Loading questions from: {qa_file}")
    all_questions = load_qa_pairs(qa_file)
    
    if not all_questions:
        print("No questions found.")
        return

    # Select first 10
    test_questions = all_questions[:10]
    print(f"Selected {len(test_questions)} questions for evaluation.")
    
    results = []
    
    for idx, item in enumerate(test_questions):
        question = item['question']
        print(f"\nProcessing Q{idx+1}: {question[:50]}...")
        
        # Add delay to avoid Rate Limit (429) on free tier
        if idx > 0:
            print("Waiting 15s to respect rate limits...")
            time.sleep(15)
        
        start_time = time.time()
        
        try:
            # Retry Logic for Rate Limits
            max_retries = 3
            for attempt in range(max_retries):
                response = rag.get_answer(question)
                result_text = response.get("result", "")
                
                if "Quota exceeded" in result_text or "429" in result_text or "429" in str(response):
                    if attempt < max_retries - 1:
                        wait = (attempt + 1) * 30 # Wait 30s, 60s
                        print(f"Rate limit hit. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        raise Exception(f"Quota Exceeded after {max_retries} attempts: {result_text}")
                
                if "Error generating answer" in result_text:
                     # Some other error
                     raise Exception(f"Generation Error: {result_text}")
                     
                break # Success
            
            end_time = time.time()
            latency = end_time - start_time
            
            token_usage = response.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            
            cost = calculate_cost(prompt_tokens, completion_tokens)
            
            # Context Count (Number of chunks retrieved)
            context_count = len(response.get('source_documents', []))
            
            result_row = {
                "Question ID": idx + 1,
                "Latency (s)": round(latency, 4),
                "Input Tokens": prompt_tokens,
                "Output Tokens": completion_tokens,
                "Total Tokens": total_tokens,
                "Cost ($)": f"{cost:.8f}",
                "Context chunks": context_count
            }
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing question: {e}")
            
    # Create Report (No Pandas)
    print("\n" + "="*85)
    print(f"{'ID':<3} | {'Latency (s)':<12} | {'In Tok':<8} | {'Out Tok':<8} | {'Tot Tok':<8} | {'Cost ($)':<12} | {'Context':<8}")
    print("-" * 85)
    
    total_cost = 0.0
    total_latency = 0.0
    total_input = 0
    total_output = 0
    total_tokens_all = 0
    
    for row in results:
        print(f"{row['Question ID']:<3} | {row['Latency (s)']:<12} | {row['Input Tokens']:<8} | {row['Output Tokens']:<8} | {row['Total Tokens']:<8} | {row['Cost ($)']:<12} | {row['Context chunks']:<8}")
        
        total_latency += row['Latency (s)']
        total_input += row['Input Tokens']
        total_output += row['Output Tokens']
        total_tokens_all += row['Total Tokens']
        total_cost += float(row['Cost ($)'])
        
    count = len(results)
    if count > 0:
        avg_latency = total_latency / count
        avg_input = total_input / count
        avg_output = total_output / count
        avg_total = total_tokens_all / count
        
        print("-" * 85)
        print("\n--- Summary ---")
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"Average Input Tokens: {avg_input:.1f}")
        print(f"Average Output Tokens: {avg_output:.1f}")
        print(f"Average Total Tokens: {avg_total:.1f}")
        print(f"Total Estimated Cost: ${total_cost:.8f}")

if __name__ == "__main__":
    evaluate_performance()
