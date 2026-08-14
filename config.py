"""
Shared configuration and consistency utilities for the Istanbul RAG chatbot.

This module is the SINGLE SOURCE OF TRUTH for the embedding model, prefixes,
paths, and thresholds. ingest.py (document side) and pipeline.py (query side)
both import get_embeddings() from here instead of constructing their own —
that's what actually guarantees "same model and version on both sides",
rather than just hoping two separate strings never drift apart.
"""

import hashlib
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/kaggle/working/proje"
DOCS_DIR_DEFAULT = "/kaggle/input/cityguide"
DB_PATH = f"{BASE_DIR}/vectorstore/db_faiss"
FINGERPRINT_PATH = f"{BASE_DIR}/vectorstore/fingerprint.json"
CHUNK_META_PATH = f"{BASE_DIR}/vectorstore/chunks_meta.jsonl"
CACHE_DB_PATH = f"{BASE_DIR}/cache/semantic_cache.sqlite"
GOLDEN_SET_PATH = f"{BASE_DIR}/eval/golden_set.jsonl"
GOLDEN_SET_DRAFT_PATH = f"{BASE_DIR}/eval/golden_set.draft.jsonl"
EVAL_RESULTS_DIR = f"{BASE_DIR}/eval/results"

# ---------------------------------------------------------------------------
# Embedding model (query side and document side MUST match — enforced by
# the fingerprint below, not just by convention)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
NORMALIZE_EMBEDDINGS = True

# ---------------------------------------------------------------------------
# Retrieval / rerank
# ---------------------------------------------------------------------------
K_RETRIEVE = 20               # bi-encoder candidate pool (recall stage) — was 6
RERANK_TOP_N = 4              # chunks actually shown to the LLM (precision stage)
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"   # multilingual, self-hosted, ~50-100ms/query on GPU
RELEVANCE_THRESHOLD = 0.15    # sigmoid score below this -> treat as "no relevant context"
                               # TUNE with eval/run_eval.py's per-category score report

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS_REWRITE = 64   # a rewritten search query is a sentence, not an essay
MAX_NEW_TOKENS_ANSWER = 300   # was 512 for every call including the rewrite — see README
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.1

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------
CACHE_SIM_THRESHOLD = 0.96    # cosine similarity to count as "the same question"
                               # TUNE by inspecting near-duplicate questions in your logs
CACHE_TTL_SECONDS = None      # e.g. 60 * 60 * 24 * 30 for a 30-day TTL; None = no expiry

# ---------------------------------------------------------------------------
# Eval gate thresholds. hard=True -> fails eval/run_eval.py's exit code.
# hard=False -> reported, but doesn't block shipping.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "retrieval_recall_at_k":  {"target": 0.80, "hard": True},
    "rerank_precision_at_n":  {"target": 0.70, "hard": True},
    "refusal_accuracy":       {"target": 0.90, "hard": True},
    "citation_rate":          {"target": 0.75, "hard": True},
    "must_contain_accuracy":  {"target": 0.80, "hard": False},
    # Latency is a SOFT gate for now on purpose: with a 4-bit 7B model served
    # via plain transformers, cold-path P95 will likely miss 2000ms on most
    # GPUs regardless of anything in this codebase — closing that gap is an
    # infra decision (serving engine, model size), not a bug. Flip hard=True
    # once you've made that call. See README "Latency" section.
    "latency_p95_ms_cold":    {"target": 2000, "hard": False},
    "latency_p95_ms_cached":  {"target": 300,  "hard": False},
}

# The system prompt is instructed to produce these phrases verbatim.
# eval/run_eval.py checks for these exact substrings to score refusal_accuracy —
# pipeline.py's prompt is built FROM these constants (not a hand-copied string)
# so the two can't silently drift apart the way the embedding model could.
REFUSAL_PHRASE_OTHER_CITY = "sadece İstanbul gezi rehberiyim"
REFUSAL_PHRASE_NO_INFO = "net bir bilgi bulunmuyor"


def resolve_docs_dir():
    """Same fallback both ingest.py and pipeline.py need: if DOCS_DIR_DEFAULT
    doesn't exist, use the first directory under /kaggle/input (Kaggle mounts
    each attached dataset as its own subfolder there). Defined once here so
    the two sides can't resolve to different directories."""
    docs_dir = DOCS_DIR_DEFAULT
    if not os.path.exists(docs_dir) and os.path.exists("/kaggle/input"):
        input_dirs = os.listdir("/kaggle/input")
        if input_dirs:
            docs_dir = os.path.join("/kaggle/input", input_dirs[0])
    return docs_dir


def get_embeddings():
    """Single construction path for the embedding model. ingest.py and
    pipeline.py both call this instead of building their own
    HuggingFaceEmbeddings — that's what makes "same model on both sides" a
    guarantee instead of a hope."""
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )


def _embedding_revision():
    """Best-effort fetch of the current model revision from the Hub, so the
    fingerprint can catch a silent upstream model update, not just a name
    match. Returns None (rather than raising) if there's no network access,
    e.g. when main.py starts up without internet."""
    try:
        from huggingface_hub import model_info
        return model_info(EMBEDDING_MODEL_NAME).sha
    except Exception:
        return None


def _corpus_hash(docs_dir):
    """Hash of (filename, size, mtime) for every source doc, so re-ingesting
    changed documents invalidates the cache/fingerprint even when the
    embedding model itself hasn't changed."""
    entries = []
    p = Path(docs_dir)
    if p.exists():
        for f in sorted(p.rglob("*.txt")):
            stat = f.stat()
            entries.append(f"{f.name}:{stat.st_size}:{int(stat.st_mtime)}")
    digest = hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()
    return digest[:16]


def compute_fingerprint(embeddings, docs_dir=None):
    """Build the fingerprint describing exactly how embeddings were produced,
    so query-side code can verify it's compatible with the index it's about
    to search before ever calling FAISS."""
    if docs_dir is None:
        docs_dir = resolve_docs_dir()
    dim = len(embeddings.embed_query("fingerprint probe"))
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": _embedding_revision(),
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "embedding_dim": dim,
        "corpus_hash": _corpus_hash(docs_dir),
    }


def save_fingerprint(fp, path=FINGERPRINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=2)


def load_fingerprint(path=FINGERPRINT_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_fingerprint(current, stored):
    """Compare the fingerprint just computed (query side) against the one
    saved at ingestion time (document side). Returns (hard_mismatches,
    soft_mismatches). Hard mismatches mean the two sides are NOT comparable
    and should block startup; soft ones are worth a warning."""
    if stored is None:
        return (["no fingerprint.json found — run ingest.py first"], [])

    hard, soft = [], []
    for key in ("embedding_model", "query_prefix", "passage_prefix",
                "normalize_embeddings", "embedding_dim"):
        if current.get(key) != stored.get(key):
            hard.append(f"{key}: query-side={current.get(key)!r} vs index={stored.get(key)!r}")

    if current.get("corpus_hash") != stored.get("corpus_hash"):
        soft.append("corpus_hash differs — source docs changed since last ingest; "
                     "re-run ingest.py so retrieval reflects the latest content")

    if (current.get("embedding_revision") and stored.get("embedding_revision")
            and current["embedding_revision"] != stored["embedding_revision"]):
        soft.append("embedding_revision differs — the embedding model was updated "
                     "on the Hub since you ingested; consider re-ingesting")

    return (hard, soft)


def index_version(fp):
    """Short id used to namespace the semantic cache, so cache entries built
    against an old index/model are never served against a new one."""
    if fp is None:
        return "unknown"
    raw = json.dumps(fp, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
