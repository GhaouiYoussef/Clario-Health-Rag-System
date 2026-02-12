import json
import time
import os
import sys
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG
from dotenv import load_dotenv

# Define the model to use for evaluation here
MODEL_NAME = "gemini-2.5-flash-lite"
COSTS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'costs_gemini.json')

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_metrics(reference_answer, generated_answer):
    """
    Calculates ROUGE and BLEU scores between reference and generated answers.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_answer, generated_answer)
    
    # Calculate BLEU score
    smooth = SmoothingFunction().method1
    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smooth)
    
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
        'bleu_score': bleu_score
    }

def load_model_config(model_name):
    """Loads cost and rpm configuration for the specified model from json file."""
    try:
        with open(COSTS_FILE_PATH, 'r') as f:
            data = json.load(f)
        
        models = data.get("models", {})
        if model_name in models:
            return models[model_name]
        else:
            print(f"Warning: Model '{model_name}' not found in costs file. Using default zero costs.")
            return {}
    except Exception as e:
        print(f"Error loading costs file: {e}")
        return {}

def estimate_cost(token_usage, model_config):
    """
    Estimates cost based on token usage and loaded configuration.
    """
    
    input_price = model_config.get("embedding_cost_per_1M_tokens", 0.0)
    output_price = model_config.get("generation_cost_per_1M_tokens", 0.0)
    
    prompt_tokens = token_usage.get('prompt_tokens', 0)
    completion_tokens = token_usage.get('completion_tokens', 0)
    
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    
    return input_cost + output_cost

def evaluate_agent(test_file_path, output_file_path):
    load_dotenv()
    
    # Load Model Configuration
    model_config = load_model_config(MODEL_NAME)
    rpm = model_config.get("rpm", 10) # Default to conservative 10 if missing
    sleep_time = 60.0 / rpm if rpm > 0 else 0
    print(f"Evaluation Configuration:")
    print(f"- Model: {MODEL_NAME}")
    print(f"- RPM Limit: {rpm} (Sleep {sleep_time:.2f}s per request)")
    
    # Choose database path - ensure it matches the one used in the app
    db_path = "./data/chroma_db_chap_based"
    if not os.path.exists(db_path):
        db_path = "./data/chroma_db_normal"
        print(f"Using fallback DB: {db_path}")
    else:
        print(f"Using DB: {db_path}")

    print("Initializing RAG Agent...")
    # Initialize your RAG agent with specific model
    rag_agent = HealthcareRAG(persist_directory=db_path, model_name=MODEL_NAME) 
    
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    results = []
    total_cost = 0
    total_latency = 0
    
    print(f"Starting evaluation on {len(test_data)} test cases...")
    
    for item in tqdm(test_data):
        question = item['question']
        reference_answer = item['answer']
        
        start_time = time.time()
        
        # Get answer from agent
        try:
            response = rag_agent.get_answer(
                question, 
                n_results=5, 
                k_candidates=40,
                doc_diversity=0.2
            )
            generated_answer = response.get('result', '')
            token_usage = response.get('token_usage', {}) or {} # Ensure dict
            
        except Exception as e:
            print(f"Error processing question: {question}. Error: {e}")
            print("Stopping evaluation due to generation error.")
            break 
            # generated_answer = "Error generating response."
            # token_usage = {}

        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(reference_answer, generated_answer)
        
        # Estimate cost using loaded config
        cost = estimate_cost(token_usage, model_config)
        
        result_entry = {
            'model': MODEL_NAME,
            'question': question,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'metrics': metrics,
            'latency_seconds': latency,
            'token_usage': token_usage,
            'estimated_cost_usd': cost,
            'source_documents': [
                m.get('metadata', {}).get('source', 'unknown') 
                for m in response.get('source_documents', [])
            ]
        }
        
        results.append(result_entry)
        total_cost += cost
        total_latency += latency
        
        # Rate limiting based on RPM
        time.sleep(sleep_time)

    # Aggregated stats

    # Aggregated stats
    latencies = [r['latency_seconds'] for r in results]
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    
    avg_rouge1 = np.mean([r['metrics']['rouge1_f1'] for r in results]) if results else 0
    avg_rougeL = np.mean([r['metrics']['rougeL_f1'] for r in results]) if results else 0
    avg_bleu = np.mean([r['metrics']['bleu_score'] for r in results]) if results else 0
    
    avg_cost_per_query = total_cost / len(test_data) if test_data else 0
    cost_per_1k_queries = avg_cost_per_query * 1000

    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_NAME,
        'config': {
            'n_results': 5,
            'k_candidates': 40,
            'doc_diversity': 0.2
        },
        'total_samples': len(test_data),
        'latency': {
            'average_seconds': avg_latency,
            'p95_seconds': p95_latency
        },
        'cost': {
            'total_usd': total_cost,
            'average_per_query_usd': avg_cost_per_query,
            'estimated_per_1k_queries_usd': cost_per_1k_queries
        },
        'quality_metrics': {
            'average_rouge1_f1': avg_rouge1,
            'average_rougeL_f1': avg_rougeL,
            'average_bleu_score': avg_bleu
        },
        'per_sample_results': results
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n=== BENCHMARK REPORT: {MODEL_NAME} ===")
    print(f"Total Samples: {len(test_data)}")
    print(f"\n--- Speed/Latency ---")
    print(f"Average Response Time: {avg_latency:.2f}s")
    print(f"P95 Response Time:     {p95_latency:.2f}s")
    print(f"\n--- Cost Efficiency ---")
    print(f"Total Cost (Run):      ${total_cost:.6f}")
    print(f"Avg Cost per Query:    ${avg_cost_per_query:.6f}")
    print(f"Cost per 1k Queries:   ${cost_per_1k_queries:.4f}")
    print(f"\n--- Accuracy/Quality ---")
    print(f"Avg ROUGE-1:           {avg_rouge1:.4f}")
    print(f"Avg ROUGE-L:           {avg_rougeL:.4f}")
    print(f"Avg BLEU:              {avg_bleu:.4f}")
    print(f"==========================================")
    print(f"Detailed results saved to: {output_file_path}")

if __name__ == "__main__":
    test_file = "./data/healthcare_golden_dataset.json"
    output_file = "./evaluation/evaluation_results.json"
    
    if os.path.exists(test_file):
        evaluate_agent(test_file, output_file)
    else:
        print(f"Test file not found: {test_file}")
