import json
import pickle

# Load chunks
with open("data/bm25_index.pkl", "rb") as f:
    index_data = pickle.load(f)
chunks_map = {c["chunk_id"]: c["text"] for c in index_data["chunks"]}

# Load eval results
with open("data/eval_results_v0.json") as f:
    results = json.load(f)

# Load queries
with open("data/eval_queries_v0.json") as f:
    queries_data = json.load(f)

print("="*80)
print("GROUND TRUTH ANNOTATION - STRICT MODE")
print("="*80 + "\n")

for result in results:
    qid = result["query_id"]
    query_text = result["query"]
    retrieved = result["retrieved_chunks"]
    
    print(f"\n{'='*80}")
    print(f"{qid}: {query_text}")
    print('='*80)
    
    for i, chunk_id in enumerate(retrieved, 1):
        chunk_text = chunks_map.get(chunk_id, "[CHUNK NOT FOUND]")
        print(f"\n[{i}] {chunk_id}")
        print("-" * 80)
        print(chunk_text)
        print("-" * 80)
    
    correct = input(f"\n✅ Enter rank numbers of CORRECT chunks (comma-separated) or 'none': ").strip()
    
    if correct.lower() != 'none' and correct:
        correct_ranks = [int(x.strip()) for x in correct.split(',')]
        correct_chunk_ids = [retrieved[r-1] for r in correct_ranks if 0 < r <= len(retrieved)]
    else:
        correct_chunk_ids = []
    
    # Update
    for q in queries_data["queries"]:
        if q["id"] == qid:
            q["ground_truth_chunks"] = correct_chunk_ids
            print(f"   Annotated: {correct_chunk_ids if correct_chunk_ids else 'NONE CORRECT'}")
            break

# Save
with open("data/eval_queries_v0.json", "w") as f:
    json.dump(queries_data, f, indent=2)

print("\n" + "="*80)
print("✅ Ground truth annotations saved to data/eval_queries_v0.json")
print("="*80)
