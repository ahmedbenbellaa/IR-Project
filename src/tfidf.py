
import math
import logging

logger = logging.getLogger(__name__)



def _build_query_vector(query_tokens: list, idf_table: dict) -> dict:
   
    # count how many times each term appears in the query
    term_counts: dict[str, int] = {}
    for token in query_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    query_vector: dict[str, float] = {}
    for term, count in term_counts.items():
        tf     = 1.0 + math.log10(count)    # dampened term frequency
        idf    = idf_table.get(term, 0.0)   # 0.0 if term not in index
        weight = tf * idf
        if weight > 0:
            query_vector[term] = weight

    return query_vector




def _cosine_similarity(
    query_vector:   dict,
    doc_id:         str,
    tf_table:       dict,
    idf_table:      dict,
    doc_norms:      dict,
) -> float:
    
    doc_tf_weights = tf_table.get(doc_id, {})
    doc_norm       = doc_norms.get(doc_id, 0.0)

    # dot product over shared terms
    dot_product = 0.0
    for term, q_weight in query_vector.items():
        doc_tf  = doc_tf_weights.get(term, 0.0)
        doc_idf = idf_table.get(term, 0.0)
        dot_product += q_weight * (doc_tf * doc_idf)

    # query L2 norm
    query_norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))

    denominator = query_norm * doc_norm
    return dot_product / denominator if denominator > 0 else 0.0



def rank(
    query_tokens:      list,
    candidate_doc_ids: list,
    index:             dict,
    top_n:             int = 10,
) -> list:
    

    if not query_tokens or not candidate_doc_ids:
        return []

    tf_table  = index.get("tf_table",  {})
    idf_table = index.get("idf_table", {})
    doc_norms = index.get("doc_norms", {})


    query_vector = _build_query_vector(query_tokens, idf_table)

    if not query_vector:
        
        logger.warning("Query vector is empty (all terms OOV). Returning unranked.")
        return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

   
    scored = []
    for doc_id in candidate_doc_ids:
        score = _cosine_similarity(query_vector, doc_id, tf_table, idf_table, doc_norms)
        scored.append((doc_id, score))

    #  sort highest first 
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored if top_n <= 0 else scored[:top_n]


#  CLI 

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python tfidf.py <index.json> <query terms...>")
        print("Example: python tfidf.py index.json climate change")
        sys.exit(1)

    index_path  = sys.argv[1]
    query_terms = sys.argv[2:]   # remaining args are the raw query words

    try:
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading index: {e}")
        sys.exit(1)

    
    results = rank(query_terms, list(idx["lang_map"].keys()), idx)

    if not results:
        print("No results.")
    else:
        print(f"\nTop results for: {' '.join(query_terms)}\n")
        for i, (doc_id, score) in enumerate(results, 1):
            bar = "█" * int(score * 20)
            print(f"  {i}. {doc_id:<25} Score: {score:.4f}  {bar}")
