"""
Streamlit chat UI. All RAG logic lives in pipeline.py, so this file is just
rendering + session state — and so it's exactly the code path
eval/run_eval.py exercises before you ship a change.
"""

import streamlit as st

import config
import pipeline

st.set_page_config(page_title="İstanbul Gezi Rehberi AI", page_icon="🇹🇷")
st.title("🇹🇷 İstanbul Gezi Rehberi AI")


@st.cache_resource
def get_resources():
    return pipeline.load_resources()


with st.spinner("Model ve veritabanı hazırlanıyor..."):
    resources = get_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latency_log_ms" not in st.session_state:
    st.session_state.latency_log_ms = []


def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(p * (len(s) - 1)), len(s) - 1)
    return s[idx]


with st.sidebar:
    st.subheader("⏱️ Performans (bu oturum)")
    log = st.session_state.latency_log_ms
    if log:
        st.metric("P50", f"{_percentile(log, 0.50):.0f} ms")
        st.metric("P95", f"{_percentile(log, 0.95):.0f} ms")
        st.caption(f"Hedef P95: < 2000 ms · {len(log)} istekten hesaplandı. "
                    f"İlk birkaç istek soğuk başlangıç nedeniyle daha yavaş olabilir.")
    else:
        st.caption("Henüz istek yok.")
    st.divider()
    st.caption(f"Önbellek: {resources.cache.stats()['entries']} kayıtlı soru")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("cache_badge"):
            st.caption(msg["cache_badge"])

if user_input := st.chat_input("İstanbul hakkında bir soru sorun..."):
    history = st.session_state.messages  # prior turns only — new turn passed separately below

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.status("Bağlam analiz ediliyor...", expanded=False) as status:
            result = pipeline.answer_question(user_input, history, resources)
            if result.from_cache:
                status.update(label="⚡ Önbellekten yanıtlandı", state="complete")
            elif result.relevance_gated:
                status.update(label="İlgili bağlam bulunamadı", state="complete")
            else:
                status.update(label="✅ Bağlam bulundu, yanıt üretildi", state="complete")

        st.write(result.answer)

        cache_badge = None
        if result.from_cache:
            cache_badge = f"⚡ Önbellekten (benzerlik: {result.cache_similarity:.3f})"
            st.caption(cache_badge)

        if result.sources:
            cited_files = {
                s["source_file"] for i, s in enumerate(result.sources, start=1)
                if i in result.cited_indices
            }
            if cited_files:
                st.caption("Kaynaklar: " + ", ".join(sorted(cited_files)))
            with st.expander("📚 Tüm Çekilen Kaynaklar (rerank sonrası)"):
                for i, s in enumerate(result.sources, start=1):
                    used = "✓ alıntılandı" if i in result.cited_indices else ""
                    st.write(f"**[{i}] {s['source_file']}** (skor: {s['score']}) {used}")
                    st.write(f"{s['text'][:250]}...")

        st.session_state.latency_log_ms.append(result.timings_ms.get("total", 0.0))

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant", "content": result.answer, "cache_badge": cache_badge,
    })
