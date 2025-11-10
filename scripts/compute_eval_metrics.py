import json

# Load annotated queries
with open("data/eval_queries_v0.json") as f:
    queries_data = json.load(f)

# Load retrieval results
with open("data/eval_results_v0.json") as f:
    results = json.load(f)

print("="*80)
print("EVALUATION METRICS - BASELINE HYBRID RETRIEVAL")
print("="*80 + "\n")

metrics = {
    "precision_at_1": [],
    "recall_at_5": [],
    "mrr": []
}

per_query_results = []

for q in queries_data["queries"]:
    qid = q["id"]
    ground_truth = set(q["ground_truth_chunks"])
    
    # Find corresponding retrieval result
    retrieved = None
    for r in results:
        if r["query_id"] == qid:
            retrieved = r["retrieved_chunks"]
            break
    
    if not retrieved or not ground_truth:
        continue
    
    # Precision@1
    p_at_1 = 1 if retrieved[0] in ground_truth else 0
    metrics["precision_at_1"].append(p_at_1)
    
    # Recall@5
    retrieved_set = set(retrieved[:5])
    recall = len(ground_truth.intersection(retrieved_set)) / len(ground_truth)
    metrics["recall_at_5"].append(recall)
    
    # MRR (Mean Reciprocal Rank)
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
        "difficulty": q["difficulty"],
        "precision_at_1": p_at_1,
        "recall_at_5": recall,
        "mrr": reciprocal_rank,
        "ground_truth_count": len(ground_truth),
        "first_correct_rank": next((i+1 for i, c in enumerate(retrieved) if c in ground_truth), None)
    })

# Aggregate metrics
avg_precision_at_1 = sum(metrics["precision_at_1"]) / len(metrics["precision_at_1"])
avg_recall_at_5 = sum(metrics["recall_at_5"]) / len(metrics["recall_at_5"])
avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])

print("📊 AGGREGATE METRICS")
print("-" * 80)
print(f"Precision@1: {avg_precision_at_1:.3f} ({sum(metrics['precision_at_1'])}/{len(metrics['precision_at_1'])} queries)")
print(f"Recall@5:    {avg_recall_at_5:.3f}")
print(f"MRR:         {avg_mrr:.3f}")
print()

# Per-query breakdown
print("📋 PER-QUERY BREAKDOWN")
print("-" * 80)
for r in per_query_results:
    status = "✅" if r["precision_at_1"] == 1 else "❌"
    print(f"{status} {r['query_id']} [{r['bucket']}] [{r['difficulty']}]")
    print(f"   P@1: {r['precision_at_1']} | R@5: {r['recall_at_5']:.2f} | MRR: {r['mrr']:.3f} | First hit: {r['first_correct_rank']}")
    print(f"   {r['query'][:80]}...")
    print()

# Identify worst performers
worst_queries = sorted(per_query_results, key=lambda x: x["mrr"])[:3]
print("🔴 WORST PERFORMING QUERIES (Lowest MRR)")
print("-" * 80)
for r in worst_queries:
    print(f"{r['query_id']}: {r['query'][:80]}...")
    print(f"   MRR: {r['mrr']:.3f} | First hit: {r['first_correct_rank']}")
    print()

# Save metrics
eval_summary = {
    "aggregate_metrics": {
        "precision_at_1": avg_precision_at_1,
        "recall_at_5": avg_recall_at_5,
        "mrr": avg_mrr
    },
    "per_query_results": per_query_results,
    "worst_queries": [q["query_id"] for q in worst_queries]
}

with open("data/eval_scores.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print("✅ Evaluation scores saved to data/eval_scores.json")
