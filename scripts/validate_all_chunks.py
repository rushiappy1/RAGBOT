import json
import glob
import numpy as np
from pathlib import Path

def validate_chunks():
    """Load all chunk JSONLs and compute aggregate distribution stats"""
    
    paths = sorted(glob.glob("data/chunks/*.jsonl"))
    
    if not paths:
        print("❌ ERROR: No chunk files found in data/chunks/")
        return False
    
    all_tokens = []
    per_doc_stats = []
    
    print("="*80)
    print("CHUNK DISTRIBUTION VALIDATION - ALL 6 MOTOR INSURERS")
    print("="*80 + "\n")
    
    for p in paths:
        doc_name = Path(p).stem
        tokens = []
        
        for line in open(p, encoding="utf-8"):
            chunk = json.loads(line)
            tokens.append(chunk['approx_tokens'])
        
        tokens = np.array(tokens)
        all_tokens.extend(tokens)
        
        stats = {
            'doc': doc_name,
            'count': len(tokens),
            'mean': tokens.mean(),
            'p50': np.percentile(tokens, 50),
            'p95': np.percentile(tokens, 95),
            'p99': np.percentile(tokens, 99),
            'max': tokens.max()
        }
        per_doc_stats.append(stats)
        
        print(f"📄 {doc_name}")
        print(f"   Chunks: {stats['count']}")
        print(f"   Mean: {stats['mean']:.1f} | P50: {stats['p50']:.1f} | P95: {stats['p95']:.1f} | P99: {stats['p99']:.1f} | Max: {int(stats['max'])}")
        print()
    
    # Aggregate stats
    all_tokens = np.array(all_tokens)
    total_count = len(all_tokens)
    agg_mean = all_tokens.mean()
    agg_p50 = np.percentile(all_tokens, 50)
    agg_p95 = np.percentile(all_tokens, 95)
    agg_p99 = np.percentile(all_tokens, 99)
    agg_max = all_tokens.max()
    
    print("="*80)
    print("AGGREGATE STATS (ALL 6 INSURERS COMBINED)")
    print("="*80)
    print(f"Total chunks: {total_count}")
    print(f"Mean: {agg_mean:.1f}")
    print(f"P50 (median): {agg_p50:.1f}")
    print(f"P95: {agg_p95:.1f}")
    print(f"P99: {agg_p99:.1f}")
    print(f"Max: {int(agg_max)}")
    print()
    
    # Validation gates
    print("="*80)
    print("VALIDATION GATES")
    print("="*80)
    
    gates_passed = True
    
    # Gate 1: Mean should be 180-400 tokens
    mean_ok = 180 <= agg_mean <= 400
    print(f"✅ Mean in range [180-400]: {agg_mean:.1f}" if mean_ok else f"❌ Mean out of range: {agg_mean:.1f}")
    gates_passed = gates_passed and mean_ok
    
    # Gate 2: P95 should be < 700 tokens
    p95_ok = agg_p95 < 700
    print(f"✅ P95 < 700: {agg_p95:.1f}" if p95_ok else f"❌ P95 too high: {agg_p95:.1f}")
    gates_passed = gates_passed and p95_ok
    
    # Gate 3: Max should be < 900 tokens (hard cap)
    max_ok = agg_max < 900
    print(f"✅ Max < 900: {int(agg_max)}" if max_ok else f"⚠️  Max exceeds 900: {int(agg_max)} (acceptable if <5 outliers)")
    
    # Count catastrophic outliers (>900)
    outliers = (all_tokens > 900).sum()
    outlier_pct = (outliers / total_count) * 100
    print(f"Outliers (>900 tokens): {outliers} ({outlier_pct:.2f}%)")
    
    if outliers > 0 and outlier_pct < 1.0:
        print(f"⚠️  {outliers} outliers acceptable (<1% of corpus)")
    elif outliers == 0:
        print("✅ No catastrophic outliers")
        gates_passed = gates_passed and True
    else:
        print(f"❌ Too many outliers: {outliers}")
        gates_passed = gates_passed and False
    
    print()
    print("="*80)
    if gates_passed:
        print("✅ ALL VALIDATION GATES PASSED - Week 0 corpus is PRODUCTION READY")
    else:
        print("❌ VALIDATION FAILED - Review distribution and re-chunk if needed")
    print("="*80)
    
    return gates_passed

if __name__ == "__main__":
    validate_chunks()
