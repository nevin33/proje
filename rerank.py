"""
Cross-encoder reranker.

The bi-encoder (E5) retrieval step is optimized for recall over the whole
corpus, not precision on one specific query — it's normal for the actual
best chunk to come back ranked #7, not #1. The reranker re-scores a small
candidate pool with a model that looks at the query and each chunk TOGETHER
(much more accurate, but too slow to run over the whole corpus), which is
exactly the gap this closes.

Default model: BAAI/bge-reranker-v2-m3 — multilingual (Turkish included),
Apache-2.0, runs locally (~50-100ms/query on GPU) so there's no added
per-query API cost or network latency. Swap RERANK_MODEL_NAME in config.py
to try an alternative (e.g. Qwen3-Reranker-0.6B for something smaller/faster,
or a hosted reranker) — this module is the only place that needs to change.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


class CrossEncoderReranker:
    def __init__(self, model_name=config.RERANK_MODEL_NAME, device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query, texts):
        """Score `query` against every string in `texts` in a single batched
        forward pass (matters for latency — one call, not a loop). Returns a
        list of floats in [0, 1] (sigmoid of the model's relevance logit),
        same order as `texts`."""
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits.view(-1).float()
        return torch.sigmoid(logits).cpu().tolist()

    def rerank(self, query, docs, top_n=config.RERANK_TOP_N):
        """docs: list of langchain Document (page_content still carries the
        "passage: " prefix from ingestion — stripped here before scoring so
        that literal prefix text can't influence relevance).

        Returns (top_docs, top_scores, best_score), sorted by reranker score
        descending. best_score is returned even when top_n truncates the
        list, so callers can apply a relevance-gate threshold on it."""
        texts = [d.page_content.replace(config.PASSAGE_PREFIX, "", 1).strip() for d in docs]
        scores = self.score(query, texts)
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        best_score = ranked[0][1] if ranked else 0.0
        top = ranked[:top_n]
        top_docs = [d for d, _ in top]
        top_scores = [s for _, s in top]
        return top_docs, top_scores, best_score
