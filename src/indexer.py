
import os
import json
import math
import logging
from collections import defaultdict
from preprocessing import preprocess, detect_language

logging.basicConfig(level=logging.INFO, format="[indexer] %(message)s")
logger = logging.getLogger(__name__)

INDEX_CACHE = "index.json"
CORPUS_DIRS = {
    "english": os.path.join("corpus", "english"),
    "arabic":  os.path.join("corpus", "arabic"),
}



def _safe_log10(x: float) -> float:
    """log10 that never divides by zero or logs 0."""
    return math.log10(x) if x > 0 else 0.0


def _read_file(path: str) -> str | None:
    """
    Read a UTF-8 text file.
    Returns None and logs a warning on any encoding / IO error.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("Skipping %s — %s", path, exc)
        return None



def build_index(corpus_dirs: dict[str, str] | None = None) -> dict:
   
    if corpus_dirs is None:
        corpus_dirs = CORPUS_DIRS

   
    for lang, directory in corpus_dirs.items():
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Corpus directory for {lang!r} not found: {directory!r}"
            )

   
    positional_index: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tf_table:  dict[str, dict[str, float]] = defaultdict(dict)
    lang_map:  dict[str, str]              = {}

    docs_loaded = 0

    for lang, directory in corpus_dirs.items():
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue

            doc_id   = filename          
            filepath = os.path.join(directory, filename)

            text = _read_file(filepath)
            if text is None:
                continue               

            lang_map[doc_id] = lang

            tokens = preprocess(text, language=lang)
            if not tokens:
                logger.info("Empty token list for %s (empty file or all stop-words)", doc_id)
                docs_loaded += 1
                continue

           
            term_counts: dict[str, int] = defaultdict(int)
            for position, term in enumerate(tokens):
                positional_index[term][doc_id].append(position)
                term_counts[term] += 1

            
            tf_table[doc_id] = {
                term: 1.0 + _safe_log10(count)
                for term, count in term_counts.items()
            }

            docs_loaded += 1

    if docs_loaded == 0:
        logger.warning("No documents were loaded. Returning empty index.")
        return _empty_index()

    N = docs_loaded  

  
    idf_table: dict[str, float] = {}
    for term, postings in positional_index.items():
        df = len(postings)                
        idf_table[term] = _safe_log10(N / df)  

   
    doc_norms: dict[str, float] = {}
    for doc_id, term_tf in tf_table.items():
        squared_sum = sum(
            (tf_weight * idf_table.get(term, 0.0)) ** 2
            for term, tf_weight in term_tf.items()
        )
        doc_norms[doc_id] = math.sqrt(squared_sum) if squared_sum > 0 else 0.0

    
    plain_index = {
        term: dict(postings)
        for term, postings in positional_index.items()
    }

    logger.info(
        "Index built: %d documents, %d unique terms.",
        docs_loaded, len(plain_index),
    )

    return {
        "positional_index": plain_index,
        "tf_table":         dict(tf_table),
        "idf_table":        idf_table,
        "doc_norms":        doc_norms,
        "lang_map":         lang_map,
    }


def _empty_index() -> dict:
    return {
        "positional_index": {},
        "tf_table":         {},
        "idf_table":        {},
        "doc_norms":        {},
        "lang_map":         {},
    }




def save_index(index: dict, path: str = INDEX_CACHE) -> None:
    """Serialise the index to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    logger.info("Index saved to %s", path)


def load_index(path: str = INDEX_CACHE) -> dict | None:
    """
    Load the index from a JSON cache file.
    Returns None if the file does not exist or is corrupt.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            index = json.load(fh)
        logger.info("Index loaded from %s", path)
        return index
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load cached index (%s). Will rebuild.", exc)
        return None


def get_or_build_index(
    corpus_dirs: dict[str, str] | None = None,
    cache_path:  str                   = INDEX_CACHE,
    force_rebuild: bool                = False,
) -> dict:
    
    if not force_rebuild:
        cached = load_index(cache_path)
        if cached is not None:
            return cached

    index = build_index(corpus_dirs)
    save_index(index, cache_path)
    return index




if __name__ == "__main__":
    import sys

    force = "--rebuild" in sys.argv
    try:
        idx = get_or_build_index(force_rebuild=force)
        print(f"Positional index terms : {len(idx['positional_index'])}")
        print(f"Documents indexed      : {len(idx['lang_map'])}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
