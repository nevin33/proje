"""
Semantic cache.

Many users ask the same handful of questions phrased ten different ways
("Ayasofya kaçta açılıyor?" / "Ayasofya'nın çalışma saatleri nedir?"). This
cache embeds each incoming query and, if it's close enough (cosine
similarity, using the SAME E5 embeddings used for retrieval) to a question
already answered, returns the stored answer instead of re-running rerank +
generation — the two most expensive steps per query, and therefore the main
lever for both cost and latency on repeat traffic.

Storage: SQLite for durability across app restarts, with vectors also held
in memory as a numpy matrix for fast similarity search. Fine up to tens of
thousands of distinct cached questions; swap in a FAISS index instead of the
numpy matrix if you outgrow that.

Cache entries are namespaced by index_version (see config.index_version) so
a re-ingested corpus or a changed embedding model can never serve stale
cached answers — see _load()'s startup purge below.
"""

from dataclasses import dataclass
from typing import Optional
import json
import os
import sqlite3
import time

import numpy as np

import config


@dataclass
class CacheHit:
    answer: str
    sources: list
    cited_indices: list
    similarity: float
    cached_at: float


class SemanticCache:
    def __init__(self, db_path=config.CACHE_DB_PATH, index_version="unknown",
                 sim_threshold=config.CACHE_SIM_THRESHOLD, ttl_seconds=config.CACHE_TTL_SECONDS):
        self.db_path = db_path
        self.index_version = index_version
        self.sim_threshold = sim_threshold
        self.ttl_seconds = ttl_seconds

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

        self._embeddings = np.zeros((0, 0), dtype=np.float32)  # (n, dim)
        self._rows = []  # parallel list of dicts: query, answer, sources, cited_indices, created_at
        self._load()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                embedding BLOB NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                cited_indices_json TEXT NOT NULL,
                index_version TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def _load(self):
        cur = self.conn.execute(
            "SELECT id, query, embedding, answer, sources_json, cited_indices_json, "
            "index_version, created_at FROM cache"
        )
        rows = cur.fetchall()
        vecs, kept, stale_ids = [], [], []
        for row in rows:
            (row_id, query, emb_blob, answer, sources_json,
             cited_json, idx_version, created_at) = row
            if idx_version != self.index_version:
                stale_ids.append(row_id)
                continue
            vecs.append(np.frombuffer(emb_blob, dtype=np.float32))
            kept.append({
                "query": query, "answer": answer,
                "sources": json.loads(sources_json),
                "cited_indices": json.loads(cited_json),
                "created_at": created_at,
            })
        if stale_ids:
            self.conn.executemany("DELETE FROM cache WHERE id = ?", [(i,) for i in stale_ids])
            self.conn.commit()
            print(f"[semantic_cache] {len(stale_ids)} eski (farklı index_version) kayıt temizlendi.")
        self._embeddings = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
        self._rows = kept

    def lookup_by_embedding(self, query_embedding) -> Optional[CacheHit]:
        if self._embeddings.shape[0] == 0:
            return None
        vec = np.asarray(query_embedding, dtype=np.float32)
        # Embeddings are already normalized at generation time (config.py),
        # so a plain dot product IS cosine similarity here.
        sims = self._embeddings @ vec
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < self.sim_threshold:
            return None
        row = self._rows[best_idx]
        if self.ttl_seconds is not None and (time.time() - row["created_at"]) > self.ttl_seconds:
            return None
        return CacheHit(
            answer=row["answer"], sources=row["sources"],
            cited_indices=row["cited_indices"], similarity=best_sim,
            cached_at=row["created_at"],
        )

    def store(self, query_text, query_embedding, answer, sources, cited_indices):
        vec = np.asarray(query_embedding, dtype=np.float32)
        now = time.time()
        self.conn.execute(
            "INSERT INTO cache (query, embedding, answer, sources_json, "
            "cited_indices_json, index_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query_text, vec.tobytes(), answer, json.dumps(sources, ensure_ascii=False),
             json.dumps(cited_indices), self.index_version, now),
        )
        self.conn.commit()
        if self._embeddings.shape[0] == 0:
            self._embeddings = vec.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, vec])
        self._rows.append({
            "query": query_text, "answer": answer, "sources": sources,
            "cited_indices": cited_indices, "created_at": now,
        })

    def stats(self):
        return {"entries": len(self._rows), "index_version": self.index_version}
