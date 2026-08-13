"""
Core RAG pipeline — shared by main.py (Streamlit UI) and eval/run_eval.py
(the ship-gate). Both call load_resources() once and then answer_question()
per turn. This is the whole point of factoring it out: if the eval called a
separate reimplementation instead of this exact code, "run the eval before
you ship" would only be testing that the reimplementation still agrees with
itself, not that the real app behaves correctly.

Flow per question:
  rewrite (skip on turn 1) -> embed once -> cache lookup (hit: return, done)
  -> retrieve K_RETRIEVE -> rerank to top_n -> relevance gate (too weak:
  return "I don't know", no generation call) -> generate, grounded only in
  the numbered context, citations required -> store in cache -> return
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS

import config
from rerank import CrossEncoderReranker
from semantic_cache import SemanticCache

CITATION_RE = re.compile(r"\[(\d+)\]")


def _build_system_prompt(context: str) -> str:
    # Built from config.REFUSAL_PHRASE_* rather than a hand-copied string, so
    # the prompt and eval/run_eval.py's refusal check can't silently drift
    # apart the way the two embedding-model definitions used to be able to.
    return f"""Sen SADECE İstanbul hakkında bilgi veren bir gezi asistanısın. İstanbul dışındaki hiçbir şehir (Ankara, İzmir, Bodrum vb.) hakkında konuşma yetkin YOKTUR. YALNIZCA TÜRKÇE konuşursun.

KATI KURALLAR:
1. BAŞKA ŞEHİR REDDİ: Kullanıcı İstanbul dışında bir yer sorarsa, KESİNLİKLE cevaplama, uydurma veya yorum yapma. SADECE şunu söyle: "Ben {config.REFUSAL_PHRASE_OTHER_CITY}. Ankara, İzmir gibi diğer şehirler hakkında bilgim yok ancak isterseniz İstanbul'da harika yerler önerebilirim."
2. İSTANBUL SORUSU AMA BİLGİ YOK: Soru İstanbul'la ilgiliyse ama BAĞLAM'da bilgi yoksa uydurma. "Kaynaklarımda bu konu hakkında {config.REFUSAL_PHRASE_NO_INFO}." de.
3. ÇEVİRİ ZORUNLULUĞU: BAĞLAM metni İngilizce olsa bile, cevabını %100 Türkçe ver. Başka alfabe veya dil kullanma.
4. SADECE BAĞLAM: Yalnızca aşağıdaki BAĞLAM içindeki bilgileri kullan. Kendi genel bilgini veya BAĞLAM dışı hiçbir bilgiyi kullanma.
5. KAYNAK GÖSTERME ZORUNLULUĞU: BAĞLAM'daki her parça başında [1], [2] gibi bir numarayla işaretlidir. Cevabındaki HER somut bilgi/iddianın hemen ardından, o bilgiyi hangi parçadan aldıysan o numarayı köşeli parantez içinde yaz. Örnek: "Ayasofya 537 yılında tamamlanmıştır [2]." Kullanmadığın parçalar için numara yazma. Numarasız hiçbir somut iddia yazma.

BAĞLAM:
{context}"""


def _build_rewrite_prompt(question: str, history: List[Dict]) -> str:
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-2:])
    return f"""<|im_start|>system
Sen bir arama motoru optimizasyon uzmanısın. Görevin, verilen sohbet geçmişini ve kullanıcının yeni sorusunu inceleyerek, bu soruyu tek başına anlaşılabilecek, bağımsız bir arama sorgusuna dönüştürmektir.
SADECE VE SADECE dönüştürülmüş arama sorgusunu TÜRKÇE olarak yaz. Soruya cevap verme. Çince veya başka bir dil KULLANMA. Kısa tut (en fazla bir cümle).
<|im_end|>
<|im_start|>user
Geçmiş:
{history_str}
Yeni Soru: {question}<|im_end|>
<|im_start|>assistant
"""


@dataclass
class Resources:
    embeddings: object
    vector_db: FAISS
    reranker: CrossEncoderReranker
    cache: SemanticCache
    generate_pipeline: object
    fingerprint: dict


@dataclass
class AnswerResult:
    answer: str
    sources: List[Dict] = field(default_factory=list)          # reranked chunks shown to the LLM
    cited_indices: List[int] = field(default_factory=list)      # [n] markers the model actually used
    retrieved_ids_pre_rerank: List[str] = field(default_factory=list)  # for retrieval-recall scoring
    from_cache: bool = False
    relevance_gated: bool = False   # True: short-circuited, nothing relevant enough to generate from
    cache_similarity: Optional[float] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)


def load_resources() -> Resources:
    embeddings = config.get_embeddings()

    vector_db = FAISS.load_local(
        config.DB_PATH, embeddings, allow_dangerous_deserialization=True
    )

    current_fp = config.compute_fingerprint(embeddings)
    stored_fp = config.load_fingerprint()
    hard_mismatches, soft_mismatches = config.verify_fingerprint(current_fp, stored_fp)
    if hard_mismatches:
        raise RuntimeError(
            "Embedding fingerprint uyuşmuyor — query-side ve index-side aynı "
            "model/versiyonu kullanmıyor:\n  - " + "\n  - ".join(hard_mismatches) +
            "\nÇözüm: ingest.py'yi yeniden çalıştırın."
        )
    for w in soft_mismatches:
        print(f"[UYARI] {w}")

    reranker = CrossEncoderReranker()
    cache = SemanticCache(index_version=config.index_version(stored_fp))

    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
    generate_pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=config.TEMPERATURE,
        repetition_penalty=config.REPETITION_PENALTY,
        do_sample=True,
    )

    return Resources(
        embeddings=embeddings, vector_db=vector_db, reranker=reranker,
        cache=cache, generate_pipeline=generate_pipeline, fingerprint=stored_fp,
    )


def _generate(res: Resources, prompt: str, max_new_tokens: int) -> str:
    out = res.generate_pipeline(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    return out.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()


def _rewrite_query(res: Resources, question: str, history: List[Dict]) -> str:
    if not history:
        return question
    prompt = _build_rewrite_prompt(question, history)
    return _generate(res, prompt, config.MAX_NEW_TOKENS_REWRITE)


def answer_question(question: str, history: List[Dict], res: Resources) -> AnswerResult:
    t0 = time.perf_counter()
    timings = {}

    def mark(name, start):
        timings[name] = (time.perf_counter() - start) * 1000

    # 1. Standalone query rewrite (skipped on turn 1 — nothing to rewrite against;
    #    capped at MAX_NEW_TOKENS_REWRITE=64, not the full 512-token answer budget)
    t = time.perf_counter()
    standalone_query = _rewrite_query(res, question, history)
    mark("rewrite", t)

    # 2. Embed once, reuse for both the cache lookup and retrieval below
    t = time.perf_counter()
    query_vec = res.embeddings.embed_query(f"{config.QUERY_PREFIX}{standalone_query}")
    mark("embed", t)

    # 3. Semantic cache lookup
    t = time.perf_counter()
    hit = res.cache.lookup_by_embedding(query_vec)
    mark("cache_lookup", t)
    if hit is not None:
        timings["total"] = (time.perf_counter() - t0) * 1000
        return AnswerResult(
            answer=hit.answer, sources=hit.sources, cited_indices=hit.cited_indices,
            from_cache=True, cache_similarity=hit.similarity, timings_ms=timings,
        )

    # 4. Retrieve a wide candidate pool (bi-encoder recall stage)
    t = time.perf_counter()
    candidates = res.vector_db.similarity_search_by_vector(query_vec, k=config.K_RETRIEVE)
    mark("retrieve", t)
    pre_rerank_ids = [d.metadata.get("chunk_id", "?") for d in candidates]

    # 5. Rerank down to the chunks actually worth showing the model (precision stage)
    t = time.perf_counter()
    top_docs, top_scores, best_score = res.reranker.rerank(
        standalone_query, candidates, top_n=config.RERANK_TOP_N
    )
    mark("rerank", t)

    sources = [
        {
            "chunk_id": d.metadata.get("chunk_id", "?"),
            "source_file": d.metadata.get("source", "unknown"),
            "text": d.page_content.replace(config.PASSAGE_PREFIX, "", 1).strip(),
            "score": round(s, 4),
        }
        for d, s in zip(top_docs, top_scores)
    ]

    # 6. Relevance gate: nothing retrieved was actually relevant enough to
    # answer from. Skip generation entirely — faster, cheaper, and avoids
    # tempting the model to answer from weak/irrelevant context.
    if best_score < config.RELEVANCE_THRESHOLD:
        answer = f"Kaynaklarımda bu konu hakkında {config.REFUSAL_PHRASE_NO_INFO}."
        timings["generate"] = 0.0
        timings["total"] = (time.perf_counter() - t0) * 1000
        res.cache.store(standalone_query, query_vec, answer, sources, [])
        return AnswerResult(
            answer=answer, sources=sources, cited_indices=[],
            retrieved_ids_pre_rerank=pre_rerank_ids,
            relevance_gated=True, timings_ms=timings,
        )

    # 7. Generate, grounded only in the numbered, reranked context
    context_block = "\n\n".join(f"[{i + 1}] {s['text']}" for i, s in enumerate(sources))
    system_prompt = _build_system_prompt(context_block)
    final_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    t = time.perf_counter()
    answer = _generate(res, final_prompt, config.MAX_NEW_TOKENS_ANSWER)
    mark("generate", t)

    cited_indices = sorted({int(n) for n in CITATION_RE.findall(answer)})
    timings["total"] = (time.perf_counter() - t0) * 1000

    res.cache.store(standalone_query, query_vec, answer, sources, cited_indices)

    return AnswerResult(
        answer=answer, sources=sources, cited_indices=cited_indices,
        retrieved_ids_pre_rerank=pre_rerank_ids, timings_ms=timings,
    )
