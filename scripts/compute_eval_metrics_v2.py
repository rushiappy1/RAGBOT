import json

# Load annotated queries
with open("data/eval_queries_v0.json") as f:
    queries_data = json.load(f)

# Load V1 (baseline) results
with open("data/eval_results_v0.json") as f:
    results_v1 = json.load(f)

# Load V2 (improved) results
with open("data/eval_results_v2.json") as f:
    results_v2 = json.load(f)

def compute_metrics(results, queries_data):
    metrics = {
        "precision_at_1": [],
        "recall_at_5": [],
        "mrr": []
    }
    
    per_query_results = []
    
    for q in queries_data["queries"]:
        qid = q["id"]
        ground_truth = set(q["ground_truth_chunks"])
        
        if not ground_truth:
            continue
        
        # Find retrieval result
        retrieved = None
        for r in results:
            if r["query_id"] == qid:
                retrieved = r["retrieved_chunks"]
                break
        
        if not retrieved:
            continue
        
        # Metrics
        p_at_1 = 1 if retrieved[0] in ground_truth else 0
        metrics["precision_at_1"].append(p_at_1)
        
        retrieved_set = set(retrieved[:5])
        recall = len(ground_truth.intersection(retrieved_set)) / len(ground_truth)
        metrics["recall_at_5"].append(recall)
        
        reciprocal_rank = 0
        for rank, chunk_id in enumerate(retrieved, 1):
            if chunk_id in ground_truth:
                reciprocal_rank = 1 / rank
                break
        metrics["mrr"].append(reciprocal_rank)
        
        per_query_results.append({
            "query_id": qid,
            "query": q["query"],
            "bucket": q["bucket"],
            "precision_at_1": p_at_1,
            "recall_at_5": recall,
            "mrr": reciprocal_rank,
            "first_correct_rank": next((i+1 for i, c in enumerate(retrieved) if c in ground_truth), None)
        })
    
    avg_precision = sum(metrics["precision_at_1"]) / len(metrics["precision_at_1"])
    avg_recall = sum(metrics["recall_at_5"]) / len(metrics["recall_at_5"])
    avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
    
    return avg_precision, avg_recall, avg_mrr, per_query_results

# Compute metrics for both versions
p1_v1, r5_v1, mrr_v1, per_query_v1 = compute_metrics(results_v1, queries_data)
p1_v2, r5_v2, mrr_v2, per_query_v2 = compute_metrics(results_v2, queries_data)

print("="*80)
print("EVALUATION COMPARISON: BASELINE (V1) vs IMPROVED (V2)")
print("="*80 + "\n")

print("📊 AGGREGATE METRICS COMPARISON")
print("-" * 80)
print(f"{'Metric':<20} | {'V1 (Baseline)':<15} | {'V2 (Improved)':<15} | {'Change':<10}")
print("-" * 80)
print(f"{'Precision@1':<20} | {p1_v1:.3f}           | {p1_v2:.3f}           | {(p1_v2-p1_v1):+.3f}")
print(f"{'Recall@5':<20} | {r5_v1:.3f}           | {r5_v2:.3f}           | {(r5_v2-r5_v1):+.3f}")
print(f"{'MRR':<20} | {mrr_v1:.3f}           | {mrr_v2:.3f}           | {(mrr_v2-mrr_v1):+.3f}")
print()

print("📋 PER-QUERY COMPARISON")
print("-" * 80)
for v1, v2 in zip(per_query_v1, per_query_v2):
    qid = v1["query_id"]
    
    # Determine status
    if v1["precision_at_1"] == 0 and v2["precision_at_1"] == 1:
        status = "✅ FIXED"
    elif v1["precision_at_1"] == 1 and v2["precision_at_1"] == 0:
        status = "❌ REGRESSION"
    elif v1["precision_at_1"] == 1 and v2["precision_at_1"] == 1:
        status = "✓ MAINTAINED"
    else:
        status = "⚠️ STILL FAILING"
    
    print(f"{status} {qid} [{v1['bucket']}]")
    print(f"   V1: P@1={v1['precision_at_1']} | MRR={v1['mrr']:.3f} | First hit: {v1['first_correct_rank']}")
    print(f"   V2: P@1={v2['precision_at_1']} | MRR={v2['mrr']:.3f} | First hit: {v2['first_correct_rank']}")
    print()

# Save comparison
comparison = {
    "v1_metrics": {"precision_at_1": p1_v1, "recall_at_5": r5_v1, "mrr": mrr_v1},
    "v2_metrics": {"precision_at_1": p1_v2, "recall_at_5": r5_v2, "mrr": mrr_v2},
    "improvements": {
        "precision_at_1_delta": p1_v2 - p1_v1,
        "mrr_delta": mrr_v2 - mrr_v1
    }
}

with open("data/eval_comparison_v1_vs_v2.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("✅ Comparison saved to data/eval_comparison_v1_vs_v2.json")
