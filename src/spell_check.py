
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


JACCARD_THRESHOLD: float = 0.2   
DEFAULT_K:         int   = 2     


def get_kgrams(word: str, k: int = DEFAULT_K) -> set[str]:
    
    if k <= 0:
        logger.warning("k must be > 0; defaulting to 2.")
        k = DEFAULT_K

    if not word:
        return set()

    if len(word) < k:
        return {word}

    return {word[i: i + k] for i in range(len(word) - k + 1)}




def jaccard_similarity(set_a: set, set_b: set) -> float:
    
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union




def levenshtein_distance(s1: str, s2: str) -> int:
   
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    m, n = len(s1), len(s2)

    # prev[j] = edit distance between s1[:i] and s2[:j]
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]               # characters match — free
            else:
                curr[j] = 1 + min(
                    prev[j],       # delete from s1
                    curr[j - 1],   # insert into s1
                    prev[j - 1],   # substitute
                )
        prev = curr

    return prev[n]




def get_suggestions(
    term:    str,
    index:   dict,
    top_n:   int = 3,
    k:       int = DEFAULT_K,
    threshold: float = JACCARD_THRESHOLD,
) -> list[str]:
    
    if not term:
        return []

    vocab: Iterable[str] = index.get("positional_index", {}).keys()


    vocab_set = set(vocab)
    if not vocab_set:
        return []

    
    if term in vocab_set:
        return []


    term_grams = get_kgrams(term, k=k)

    candidates: list[str] = []
    for known_term in vocab_set:
        known_grams = get_kgrams(known_term, k=k)
        score = jaccard_similarity(term_grams, known_grams)
        if score >= threshold:
            candidates.append(known_term)

    if not candidates:
        # No candidates survived the filter
        return []


    ranked = sorted(
        candidates,
        key=lambda candidate: (levenshtein_distance(term, candidate), candidate),
    )

    return ranked[:top_n] if top_n > 0 else ranked



if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python spell_check.py <index.json> <misspelled_term>")
        sys.exit(1)

    index_path, query_term = sys.argv[1], sys.argv[2]

    try:
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading index: {e}")
        sys.exit(1)

    suggestions = get_suggestions(query_term, idx)
    if suggestions:
        print(f"Did you mean: {' | '.join(suggestions)}")
    else:
        print("No suggestion available.")   
