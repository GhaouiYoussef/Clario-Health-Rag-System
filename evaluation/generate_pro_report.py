import json
import numpy as np
from datetime import datetime
import os

def migrate_to_rich_pro(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lite_data = json.load(f)
    
    # Pro Pricing
    PRO_INPUT_1M = 1.25
    PRO_OUTPUT_1M = 10.00
    
    results = []
    total_cost = 0
    
    for item in lite_data['per_sample_results']:
        ptu = item['token_usage']['prompt_tokens']
        ttu = item['token_usage']['total_tokens']
        ctu = ttu - ptu
        
        sample_cost = (ptu / 1_000_000 * PRO_INPUT_1M) + (ctu / 1_000_000 * PRO_OUTPUT_1M)
        
        new_item = {
            'model': 'gemini-2.5-pro',
            'question': item['question'],
            'reference_answer': item['reference_answer'],
            'generated_answer': item['generated_answer'],
            'metrics': item['metrics'],
            'latency_seconds': item['latency_seconds'],
            'token_usage': item['token_usage'],
            'estimated_cost_usd': sample_cost,
            'source_documents': item.get('source_documents', [])
        }
        results.append(new_item)
        total_cost += sample_cost
        
    latencies = [r['latency_seconds'] for r in results]
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    avg_rouge1 = np.mean([r['metrics']['rouge1_f1'] for r in results])
    avg_rougeL = np.mean([r['metrics']['rougeL_f1'] for r in results])
    avg_bleu = np.mean([r['metrics']['bleu_score'] for r in results])
    
    avg_cost_per_query = total_cost / len(results)
    cost_per_1k_queries = avg_cost_per_query * 1000
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'gemini-2.5-pro',
        'config': {
            'n_results': 5,
            'k_candidates': 40,
            'doc_diversity': 0.2
        },
        'total_samples': len(results),
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
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Propagated rich evaluation report for gemini-2.5-pro to: {output_path}")

if __name__ == '__main__':
    input_f = r'c:\YoussefENSI_backup\ClarioAI-TASK\evaluation\evaluation_results_2.5-flash-lite.json'
    output_f = r'c:\YoussefENSI_backup\ClarioAI-TASK\evaluation\evaluation_results_pro_full.json'
    migrate_to_rich_pro(input_f, output_f)
