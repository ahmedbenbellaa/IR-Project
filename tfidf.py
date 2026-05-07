# tfidf.py
# Ranks a list of candidate documents against a query
# using TF-IDF weighted vectors and Cosine Similarity.

import math
from preprocessing import preprocess


# ═════════════════════════════════════════════════════════════
# COSINE SIMILARITY
# ═════════════════════════════════════════════════════════════

def cosine_similarity(query_vec, doc_vec, doc_norm):
    """
    Compute cosine similarity between a query vector and a document vector.

    Both vectors are dicts: { term: weight }

    Formula:  cos(q, d) = dot(q, d) / (|q| * |d|)

    Edge cases handled:
    - Zero query norm   → return 0.0 (avoid division by zero)
    - Zero doc norm     → return 0.0
    - Empty vectors     → return 0.0
    - No shared terms   → dot product = 0 → return 0.0
    """
    if not query_vec or not doc_vec:
        return 0.0

    # Dot product (sum over shared terms only)
    dot = sum(
        query_vec.get(term, 0.0) * doc_vec.get(term, 0.0)
        for term in query_vec
    )

    # Query L2 norm
    query_norm = math.sqrt(sum(w ** 2 for w in query_vec.values()))

    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0

    return dot / (query_norm * doc_norm)


# ═════════════════════════════════════════════════════════════
# QUERY VECTOR BUILDER
# ═════════════════════════════════════════════════════════════

def build_query_vector(query_tokens, index):
    """
    Build the TF-IDF weight vector for the query.

    Uses the same TF formula as documents: 1 + log10(count).
    IDF values come from the pre-built index.

    Returns dict { term: tfidf_weight }

    Edge cases handled:
    - Empty token list  → return {}
    - Term not in index → IDF = 0 (term has no discriminating power)
    """
    if not query_tokens:
        return {}

    idf_table = index.get("idf", {})

    # Count raw term frequencies in the query
    raw_tf = {}
    for term in query_tokens:
        raw_tf[term] = raw_tf.get(term, 0) + 1

    query_vec = {}
    for term, count in raw_tf.items():
        tf  = 1 + math.log10(count) if count > 0 else 0
        idf = idf_table.get(term, 0)   # 0 if term unknown
        query_vec[term] = tf * idf

    return query_vec


# ═════════════════════════════════════════════════════════════
# RANKING ENTRY POINT
# ═════════════════════════════════════════════════════════════

def rank_results(candidate_doc_ids, query_tokens, index, top_n=10):
    """
    Score and rank a list of candidate documents against the query.

    Parameters
    ----------
    candidate_doc_ids : list of str  – output of searcher.search()["matches"]
    query_tokens      : list of str  – processed query tokens
    index             : dict         – the full index from indexer.get_index()
    top_n             : int          – maximum number of results to return

    Returns
    -------
    list of (doc_id, score) tuples, sorted by score descending.

    Edge cases handled:
    - Empty candidate list  → return []
    - Empty query tokens    → return [] (cannot score without a query)
    - top_n ≤ 0             → return all results
    - Doc not in tf table   → score = 0.0 (still returned but ranked last)
    - All scores = 0        → return candidates with score 0 (no results hidden)
    """
    if not candidate_doc_ids:
        return []

    if not query_tokens:
        print("[TFIDF] No query tokens to rank against.")
        return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

    tf_table   = index.get("tf",        {})
    doc_norms  = index.get("doc_norms", {})
    query_vec  = build_query_vector(query_tokens, index)

    if not query_vec:
        print("[TFIDF] Query vector is all zeros. Returning unranked results.")
        return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

    scored = []
    for doc_id in candidate_doc_ids:
        doc_vec  = tf_table.get(doc_id, {})
        doc_norm = doc_norms.get(doc_id, 0.0)
        score    = cosine_similarity(query_vec, doc_vec, doc_norm)
        scored.append((doc_id, score))

    # Sort descending by score; break ties alphabetically by doc_id
    scored.sort(key=lambda x: (-x[1], x[0]))

    if top_n and top_n > 0:
        return scored[:top_n]

    return scored


# ═════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (run: python tfidf.py)
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indexer  import get_index
    from searcher import search

    print("=" * 55)
    print("TF-IDF Ranker — Self Test")
    print("=" * 55)

    try:
        idx = get_index()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    test_queries = [
        "climate change",
        "economy",
        "تغير المناخ",
        "",               # edge: empty
        "xyznotaword",    # edge: OOV
    ]

    for q in test_queries:
        result = search(q, idx)
        ranked = rank_results(result["matches"], result["query_tokens"], idx)

        print(f"\nQuery  : {repr(q)}")
        if ranked:
            for rank, (doc_id, score) in enumerate(ranked, 1):
                print(f"  {rank}. {doc_id}  |  Score: {score:.4f}")
        else:
            print("  No results.")
