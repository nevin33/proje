"""
Ship-gate eval harness.

Runs every item in golden_set.jsonl through the SAME pipeline.answer_question
used by main.py (see pipeline.py's docstring for why that matters), scores
retrieval / refusal / citation behavior, and reports latency percentiles for
both a cold pass and a repeat pass (to show the semantic cache actually
working — same questions, second time through).

Exits 0 if every HARD threshold in config.THRESHOLDS passes, 1 otherwise.
Wire this into whatever "before it ships" means for you: a manual check
before re-launching Streamlit, a pre-commit hook, or CI (see README.md for a
sample GitHub Actions workflow).

Usage:
    python eval/run_eval.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pipeline


def load_golden_set():
    if not os.path.exists(config.GOLDEN_SET_PATH):
        print(f"HATA: {config.GOLDEN_SET_PATH} bulunamadı.")
        print("Önce şunu çalıştırın: python eval/build_golden_set.py")
        sys.exit(1)
    items = []
    with open(config.GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(p * (len(s) - 1)), len(s) - 1)
    return s[idx]


def run_pass(items, res, label):
    print(f"\n--- {label} ---")
    results = []
    for item in items:
        result = pipeline.answer_question(item["question"], [], res)
        results.append((item, result))
        tag = "cache" if result.from_cache else ("gated" if result.relevance_gated else "generated")
        print(f"  [{tag:>9}] {result.timings_ms.get('total', 0):>6.0f}ms  {item['question'][:60]}")
    return results


def is_latency_metric(key):
    return "latency" in key


def build_report(cold_results, warm_results):
    pre_rerank_hits, post_rerank_hits = [], []
    refusal_scores, citation_scores, must_contain_scores = [], [], []

    for item, result in cold_results:
        expected = set(item.get("expected_source_chunk_ids", []))
        if expected:
            pre_rerank_hits.append(bool(expected & set(result.retrieved_ids_pre_rerank)))
            post_rerank_hits.append(bool(expected & {s["chunk_id"] for s in result.sources}))

        if item["type"] == "out_of_scope_city":
            refusal_scores.append(config.REFUSAL_PHRASE_OTHER_CITY in result.answer)
        elif item["type"] == "no_info_in_corpus":
            refusal_scores.append(config.REFUSAL_PHRASE_NO_INFO in result.answer)

        if item["type"] == "in_scope":
            citation_scores.append(len(result.cited_indices) > 0)

        must_contain = item.get("must_contain") or []
        if must_contain:
            must_contain_scores.append(
                all(term.lower() in result.answer.lower() for term in must_contain)
            )

    cold_latencies = [r.timings_ms.get("total", 0) for _, r in cold_results if not r.from_cache]
    warm_latencies = [r.timings_ms.get("total", 0) for _, r in warm_results if r.from_cache]
    cache_hit_rate = (sum(1 for _, r in warm_results if r.from_cache) / len(warm_results)
                       if warm_results else None)

    def rate(scores):
        return (sum(scores) / len(scores)) if scores else None

    return {
        "retrieval_recall_at_k": rate(pre_rerank_hits),
        "rerank_precision_at_n": rate(post_rerank_hits),
        "refusal_accuracy": rate(refusal_scores),
        "citation_rate": rate(citation_scores),
        "must_contain_accuracy": rate(must_contain_scores),
        "latency_p95_ms_cold": percentile(cold_latencies, 0.95),
        "latency_p95_ms_cached": percentile(warm_latencies, 0.95),
        "cache_hit_rate_on_repeat": cache_hit_rate,
        "n_items": len(cold_results),
        "n_scored_retrieval": len(pre_rerank_hits),
        "n_scored_refusal": len(refusal_scores),
        "n_scored_citation": len(citation_scores),
    }


def print_report(report):
    print("\n=== SONUÇLAR ===")
    for key, cfg in config.THRESHOLDS.items():
        actual = report.get(key)
        target = cfg["target"]
        hard = cfg["hard"]
        if actual is None:
            print(f"  {key:<26} n/a  (golden set'te bu metriği ölçecek öğe yok)")
            continue
        if is_latency_metric(key):
            ok, cmp = actual <= target, "<="
        else:
            ok, cmp = actual >= target, ">="
        status = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"  {key:<26} {actual:>9.3f}  (hedef {cmp} {target}, {'hard' if hard else 'soft'})  [{status}]")
    print(f"\n  n_items: {report['n_items']}  |  cache_hit_rate_on_repeat: {report['cache_hit_rate_on_repeat']}")
    print(f"  örneklem -> retrieval: {report['n_scored_retrieval']}, "
          f"refusal: {report['n_scored_refusal']}, citation: {report['n_scored_citation']}")


def gate(report):
    failed = []
    for key, cfg in config.THRESHOLDS.items():
        if not cfg["hard"]:
            continue
        actual = report.get(key)
        if actual is None:
            continue
        ok = (actual <= cfg["target"]) if is_latency_metric(key) else (actual >= cfg["target"])
        if not ok:
            failed.append(key)
    return failed


def main():
    items = load_golden_set()
    print(f"{len(items)} golden set öğesi yüklendi.")

    print("Kaynaklar yükleniyor (model + vektör db)... bu biraz sürebilir.")
    res = pipeline.load_resources()

    cold_results = run_pass(items, res, "1. Geçiş (soğuk)")
    warm_results = run_pass(items, res, "2. Geçiş (aynı sorular tekrar — önbellek beklenir)")

    report = build_report(cold_results, warm_results)
    print_report(report)

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EVAL_RESULTS_DIR, f"{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Rapor kaydedildi: {out_path}")

    failed = gate(report)
    if failed:
        print(f"\n❌ GATE BAŞARISIZ: {', '.join(failed)}")
        sys.exit(1)
    print("\n✅ GATE GEÇTİ — değişikliği göndermek güvenli.")
    sys.exit(0)


if __name__ == "__main__":
    main()
