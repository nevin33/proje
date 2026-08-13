"""
Stage 1 — Ingestion.

Builds the FAISS index from the corpus and writes:
  - the FAISS index itself (config.DB_PATH)
  - fingerprint.json describing exactly how embeddings were produced, so
    pipeline.py can verify query-side and document-side actually match
    before it ever calls FAISS
  - chunks_meta.jsonl mapping a stable chunk_id -> source file + text, used
    by eval/build_golden_set.py (needs real chunks to draft questions from)
    and by the eval harness (retrieval-recall needs stable ids to check
    against)

Run this whenever the source documents change, or whenever
EMBEDDING_MODEL_NAME / chunking settings change in config.py — then run
eval/run_eval.py before you point main.py at the new index.
"""

import json
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import config


def load_and_split(docs_dir):
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        recursive=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Toplam {len(documents)} adet TXT belgesi okundu.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Stable, human-readable chunk ids: <source-file-stem>_<index>. Needed so
    # the eval harness can check "was the expected chunk retrieved", and so
    # citations in the UI can be traced back to a specific chunk/file.
    per_source_counter = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        stem = os.path.splitext(os.path.basename(source))[0]
        idx = per_source_counter.get(stem, 0)
        chunk.metadata["chunk_id"] = f"{stem}_{idx:04d}"
        per_source_counter[stem] = idx + 1

    return chunks


def main():
    docs_dir = config.resolve_docs_dir()
    print(f"Kullanılan klasör: {docs_dir}")

    if not os.path.exists(docs_dir):
        print(f"UYARI: {docs_dir} bulunamadı, ingestion atlanıyor.")
        return

    chunks = load_and_split(docs_dir)

    embeddings = config.get_embeddings()

    # Save chunk metadata BEFORE embedding, so build_golden_set.py can sample
    # from it even if the FAISS build step below is ever re-run separately.
    os.makedirs(os.path.dirname(config.CHUNK_META_PATH), exist_ok=True)
    with open(config.CHUNK_META_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "chunk_id": chunk.metadata["chunk_id"],
                "source": chunk.metadata.get("source", "unknown"),
                "text": chunk.page_content,
            }, ensure_ascii=False) + "\n")
    print(f"Chunk metadata kaydedildi: {config.CHUNK_META_PATH} ({len(chunks)} parça)")

    # E5 convention: documents get "passage: ", queries get "query: ".
    for chunk in chunks:
        chunk.page_content = f"{config.PASSAGE_PREFIX}{chunk.page_content}"

    vector_db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    vector_db.save_local(config.DB_PATH)
    print(f"Vektör veritabanı kaydedildi: {config.DB_PATH}")

    fp = config.compute_fingerprint(embeddings, docs_dir=docs_dir)
    config.save_fingerprint(fp)
    print(f"Fingerprint kaydedildi: {config.FINGERPRINT_PATH}")
    print(json.dumps(fp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
