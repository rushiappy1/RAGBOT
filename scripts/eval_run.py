import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.answer_synthesis import synthesize_answer

def run_eval():
    """Run evaluation on golden dataset"""
    
    # Load eval queries
    eval_file = Path("data/eval_queries.json")
    evals = json.loads(eval_file.read_text())
    
    results = []
    for item in evals:
        q = item["query"]
        gold = set(item["gold_chunks"])
        
        print(f"\nEvaluating: {item['id']}")
        
        # Run synthesis
        try:
            out = synthesize_answer(q, debug=False)
            retrieved_ids = [f"{c['doc_id']}::{c['chunk_id']}" for c in out["retrieved_chunks"]]
            
            # Compute metrics
            p1 = 1 if retrieved_ids and any(rid in gold for rid in [retrieved_ids[0]]) else 0
            
            # MRR
            rr = 0
            for i, rid in enumerate(retrieved_ids, 1):
                if rid in gold:
                    rr = 1.0 / i
                    break
            
            # Recall@5
            r_at_5 = 1 if any(rid in gold for rid in retrieved_ids[:5]) else 0
            
            # Check expected answer
            answer_correct = (
                item["expected_answer"].lower() in out["answer"].lower()
                if item["expected_answer"] != "Not found"
                else "not found" in out["answer"].lower()
            )
            
            results.append({
                "id": item["id"],
                "query": q,
                "p1": p1,
                "rr": rr,
                "r5": r_at_5,
                "answer_correct": int(answer_correct),
                "retrieved": retrieved_ids[:5],
                "answer": out["answer"][:100] + "..."
            })
            
            print(f"  P@1: {p1}, MRR: {rr:.2f}, R@5: {r_at_5}, Answer: {'✓' if answer_correct else '✗'}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": item["id"],
                "query": q,
                "error": str(e)
            })
    
    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    
    p1_avg = sum(r["p1"] for r in valid_results) / len(valid_results) if valid_results else 0
    mrr_avg = sum(r["rr"] for r in valid_results) / len(valid_results) if valid_results else 0
    r5_avg = sum(r["r5"] for r in valid_results) / len(valid_results) if valid_results else 0
    answer_accuracy = sum(r["answer_correct"] for r in valid_results) / len(valid_results) if valid_results else 0
    
    summary = {
        "P@1": round(p1_avg, 3),
        "MRR": round(mrr_avg, 3),
        "R@5": round(r5_avg, 3),
        "Answer_Accuracy": round(answer_accuracy, 3),
        "Total_Queries": len(results),
        "Successful": len(valid_results),
        "Failed": len(results) - len(valid_results)
    }
    
    output = {
        "summary": summary,
        "details": results
    }
    
    # Save results
    Path("eval/eval_results.json").write_text(json.dumps(output, indent=2))
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for k, v in summary.items():
        print(f"{k:20s}: {v}")
    print("="*80)
    
    return output

if __name__ == "__main__":
    run_eval()
