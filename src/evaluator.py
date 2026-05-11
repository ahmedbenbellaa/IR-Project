

from searcher import search
from tfidf    import rank



# METRICS

def precision(retrieved, relevant):
  
    if not retrieved:
        return 0.0
    if not relevant:
        return 0.0

    retrieved = set(retrieved)
    relevant  = set(relevant)

    return len(retrieved & relevant) / len(retrieved)


def recall(retrieved, relevant):
  
    if not relevant:
        return 0.0
    if not retrieved:
        return 0.0

    retrieved = set(retrieved)
    relevant  = set(relevant)

    return len(retrieved & relevant) / len(relevant)


def f1_score(p, r):
 
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)




GROUND_TRUTH = {
    "climate change": {
        "relevant": ["doc_en_01.txt", "doc_en_10.txt"],
        "language": "english",
    },
    "artificial intelligence": {
        "relevant": ["doc_en_02.txt"],
        "language": "english",
    },
    "economy": {
        "relevant": ["doc_en_03.txt"],
        "language": "english",
    },
    "health": {
        "relevant": ["doc_en_05.txt"],
        "language": "english",
    },
    # Arabic query — at least one required by the project spec
    "تغير المناخ": {
        "relevant": ["doc_ar_01.txt", "doc_ar_10.txt"],
        "language": "arabic",
    },
}



# EVALUATION RUNNER


def evaluate(index, ground_truth=None, top_n=10):
 
    if ground_truth is None:
        ground_truth = GROUND_TRUTH

    if not ground_truth:
        print("[EVAL] Ground truth is empty. Nothing to evaluate.")
        return []

    results = []

    print("\n" + "=" * 65)
    print("  Search Engine Evaluation Report")
    print("=" * 65)

    for query_str, meta in ground_truth.items():
        relevant = set(meta.get("relevant", []))
        language = meta.get("language", None)

        if not relevant:
            print(f"[EVAL] WARNING: No relevant docs defined for '{query_str}'. Skipping.")
            continue

        # Run the search
        search_result = search(query_str, index, language=language)
        ranked        = rank(
            search_result["query_tokens"],
            search_result["matches"],
            index,
            top_n=top_n,
        )
        retrieved = set(doc_id for doc_id, _ in ranked)

        # Compute metrics
        p  = precision(retrieved, relevant)
        r  = recall(retrieved, relevant)
        f1 = f1_score(p, r)

        results.append({
            "query":     query_str,
            "retrieved": retrieved,
            "relevant":  relevant,
            "precision": p,
            "recall":    r,
            "f1":        f1,
        })

        # ── Print per-query report 
        print(f"\n  Query     : {query_str}")
        print(f"  Language  : {search_result['language']}")
        print(f"  Retrieved : {sorted(retrieved) if retrieved else '(none)'}")
        print(f"  Relevant  : {sorted(relevant)}")
        print(f"  Precision : {p:.3f}")
        print(f"  Recall    : {r:.3f}")
        print(f"  F1 Score  : {f1:.3f}")

        if search_result["missing_terms"]:
            print(f"  OOV terms : {search_result['missing_terms']}")

    # ── Macro averages 
    if results:
        avg_p  = sum(r["precision"] for r in results) / len(results)
        avg_r  = sum(r["recall"]    for r in results) / len(results)
        avg_f1 = sum(r["f1"]        for r in results) / len(results)

        print("\n" + "-" * 65)
        print(f"  Macro-average Precision : {avg_p:.3f}")
        print(f"  Macro-average Recall    : {avg_r:.3f}")
        print(f"  Macro-average F1        : {avg_f1:.3f}")
        print("=" * 65)

    return results



#demo
if __name__ == "__main__":
    from indexer import get_or_build_index

    print("=" * 55)
    print("Evaluator — Self Test")
    print("=" * 55)

    # Unit tests for metrics
    print("\n── Metric Unit Tests ────────────────────────────────")
    tests = [
        # (retrieved,         relevant,           exp_p, exp_r)
        (["a","b","c"],    ["a","b"],            2/3,   1.0),
        (["a","b"],        ["a","b","c"],        1.0,   2/3),
        ([],               ["a"],                0.0,   0.0),
        (["a"],            [],                   0.0,   0.0),
        (["x","y"],        ["a","b"],            0.0,   0.0),
    ]
    all_pass = True
    for ret, rel, exp_p, exp_r in tests:
        p = precision(ret, rel)
        r = recall(ret, rel)
        ok = abs(p - exp_p) < 1e-9 and abs(r - exp_r) < 1e-9
        status = "✓" if ok else "✗"
        print(f"  {status}  precision={p:.3f} (exp {exp_p:.3f})  "
            f"recall={r:.3f} (exp {exp_r:.3f})")
        if not ok:
            all_pass = False
    print("  All metric tests passed!" if all_pass else "  Some tests FAILED.")

    # Integration test
    print("\n── Full Evaluation (requires built index) ───────────")
    try:
        idx = get_or_build_index()
        evaluate(idx)
    except FileNotFoundError:
        print("  [SKIP] Index not found — run fetch_corpus.py and indexer.py first.")
