import json

# Load eval results
with open("data/eval_results_v0.json") as f:
    results = json.load(f)

# Load queries
with open("data/eval_queries_v0.json") as f:
    queries_data = json.load(f)

print("="*80)
print("GROUND TRUTH ANNOTATION HELPER")
print("="*80 + "\n")
print("For each query, review the top-5 chunks and mark which are correct.\n")

for result in results:
    qid = result["query_id"]
    query_text = result["query"]
    retrieved = result["retrieved_chunks"]
    
    print(f"\n{qid}: {query_text}")
    print("-" * 80)
    print("Retrieved chunks:")
    for i, chunk_id in enumerate(retrieved, 1):
        print(f"  [{i}] {chunk_id}")
    
    correct = input("\nEnter rank numbers of CORRECT chunks (comma-separated, or 'none'): ").strip()
    
    if correct.lower() != 'none':
        correct_ranks = [int(x.strip()) for x in correct.split(',')]
        correct_chunk_ids = [retrieved[r-1] for r in correct_ranks]
    else:
        correct_chunk_ids = []
    
    # Update queries data
    for q in queries_data["queries"]:
        if q["id"] == qid:
            q["ground_truth_chunks"] = correct_chunk_ids
            break

# Save updated queries
with open("data/eval_queries_v0.json", "w") as f:
    json.dump(queries_data, f, indent=2)

print("\n✅ Ground truth annotations saved to data/eval_queries_v0.json")
