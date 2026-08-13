# İstanbul RAG — hardened build

Same chatbot, restructured so every requirement is a real, testable piece of
the system rather than a prompt-only promise.

## Files

| File | Role |
|---|---|
| `config.py` | Single source of truth: embedding model, paths, thresholds, fingerprinting |
| `ingest.py` | Builds the FAISS index, saves fingerprint + chunk metadata |
| `rerank.py` | Cross-encoder reranker (BGE-reranker-v2-m3) |
| `semantic_cache.py` | SQLite + in-memory cosine cache for repeated questions |
| `pipeline.py` | The actual RAG logic — rewrite → retrieve → rerank → gate → generate |
| `main.py` | Streamlit UI. Thin — calls `pipeline.py`, renders the result |
| `eval/build_golden_set.py` | Drafts candidate Q&A from your real chunks + fixed OOD tests |
| `eval/run_eval.py` | The ship-gate: runs `golden_set.jsonl` through `pipeline.py`, scores it, exits non-zero on failure |

`main.py` and `eval/run_eval.py` both call `pipeline.answer_question()` —
the same function, not two implementations that happen to agree today. That's
what makes "every change runs against the eval before it ships" mean
something: the eval is exercising the exact code your users hit.

## Run order

```bash
pip install -r requirements.txt

# 1. Build the index (run again whenever docs or config.py's embedding
#    settings change)
python ingest.py

# 2. Draft a golden set from your real corpus
python eval/build_golden_set.py --n 20
# -> review eval/golden_set.draft.jsonl against source_text_preview,
#    fix anything wrong, move good items into eval/golden_set.jsonl
#    (which already has the fixed refusal/OOD cases)

# 3. Gate: run before every change you ship
python eval/run_eval.py
# exit code 0 = safe to ship, 1 = a hard threshold failed, see the printed report

# 4. Serve (unchanged from your original ngrok/streamlit launch — just point
#    it at this main.py)
streamlit run main.py --server.port 8501 --server.headless true
```

If you move this into a git repo, wire step 3 into CI, e.g.:

```yaml
# .github/workflows/rag-eval-gate.yml
name: RAG eval gate
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest   # needs GPU access for the LLM step — self-hosted runner or similar
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python eval/run_eval.py
```

## What maps to what

| You asked for | What's here |
|---|---|
| No eval set / no golden Q&A | `build_golden_set.py` auto-drafts from real chunks (needs human review) + 6 fixed OOD/refusal cases (safe to use as-is) |
| Every change runs against it before it ships | `run_eval.py`, real pipeline, hard-fails the gate on regression |
| Retrieval recall not solid | `K_RETRIEVE=20` (was 6) feeds the reranker a wider pool |
| Rerank the top hits | `rerank.py`, BGE-reranker-v2-m3, batched single forward pass |
| Model sees top chunks | Post-rerank top 4, numbered, not raw bi-encoder top-k |
| Stop making things up / context-only | Stronger prompt + a relevance gate that skips generation (returns "I don't know") when the best rerank score is below `RELEVANCE_THRESHOLD` |
| Cite sources | Numbered context, required `[n]` markers, parsed and shown in the UI |
| Say "I don't know" for irrelevant | Reinforced prompt + directly tested by the OOD golden items (`refusal_accuracy`) |
| Same embedding model+version both sides | `config.get_embeddings()` is the only construction path; a fingerprint is saved at ingest and verified at load — mismatches hard-fail startup, not silently degrade |
| P95 < 2s | See below — built what's controllable, flagged what isn't |
| Cheap repeats via semantic cache | `semantic_cache.py`, cosine over the same E5 embeddings, skips rerank + generation on a hit |

## Latency — the honest part

Hitting P95 < 2s on the **cold** path (no cache hit) is unlikely with this
exact stack on typical hardware. A 4-bit 7B model generating a few hundred
tokens through plain `transformers.pipeline` (no continuous batching, no
paged KV-cache) commonly takes several seconds per call on its own — and the
original flow made *two* such calls per question (rewrite + answer).

What this build does to shrink that:
- Rewrite is capped at 64 tokens instead of sharing the 512-token budget that was applied to every call.
- Answers are capped at 300 tokens (was 512).
- The relevance gate skips generation entirely when nothing relevant was retrieved.
- The semantic cache skips rerank *and* generation entirely on a hit — this is the only piece here that can plausibly get repeat traffic under 2s.
- Per-stage timing is recorded on every request (`timings_ms` in `AnswerResult`), and `main.py`'s sidebar shows a running P50/P95 so you can see where the time actually goes on your GPU.

What it doesn't do, because it's an infrastructure decision rather than a
code fix:
- **Swap the serving engine.** Plain `transformers` generation is the slowest common way to serve a 7B model. vLLM or TGI (paged attention, continuous batching) is usually the single biggest lever — often several times faster — and is the first thing I'd change if 2s still isn't met after warming the cache.
- **Use a smaller model.** Qwen2.5-3B or 1.5B-Instruct would very likely clear 2s on plain `transformers` where the 7B won't, trading some answer quality.

Because of this, `latency_p95_ms_cold` in `config.THRESHOLDS` is a **soft**
gate (reported, doesn't block `run_eval.py`) until you've made one of those
two calls. `latency_p95_ms_cached` is soft too, but should come in low —
if it doesn't, that's a real bug worth chasing, not a hardware ceiling.

## Tunable knobs (in `config.py`)

| Knob | Default | How to tune it |
|---|---|---|
| `RELEVANCE_THRESHOLD` | 0.15 | `run_eval.py`'s per-item output includes rerank scores — compare the distribution for `in_scope` vs OOD items and pick a cutoff that separates them |
| `CACHE_SIM_THRESHOLD` | 0.96 | Inspect near-duplicate questions in real logs; too low merges genuinely different questions, too high defeats the cache |
| `K_RETRIEVE` / `RERANK_TOP_N` | 20 / 4 | Raise `K_RETRIEVE` if `retrieval_recall_at_k` is still weak after reranking; raise `RERANK_TOP_N` if the model seems to be missing context it needs |
| `config.THRESHOLDS` | see file | Each metric is independently hard/soft — flip `latency_p95_ms_cold` to hard once your serving backend can hit it |

## Known limitations (v1)

- The golden set is single-turn — the query-rewrite path isn't exercised by `run_eval.py`. Worth adding a `history` field to golden items if multi-turn regressions become a concern.
- Scoring is deliberately lexical/deterministic (substring and set-overlap checks), not an LLM judge — cheaper, faster, and reproducible, but won't catch a paraphrased-but-wrong answer that happens to cite the right chunk. An LLM-as-judge pass is a reasonable v2 addition if you need finer-grained correctness scoring.
- `must_contain_accuracy` only means anything once you've filled in that field on reviewed golden items — it starts as a no-op.

## One more thing

Your `NGROK_AUTH_TOKEN` is hardcoded in plaintext in the original notebook
cell. If this notebook has ever been shared, committed, or made public,
treat that token as compromised and rotate it — otherwise, worth moving to
Kaggle Secrets or an environment variable (`os.environ["NGROK_AUTH_TOKEN"]`)
so it doesn't end up in version history or a screenshot.
