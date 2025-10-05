from __future__ import annotations
from typing import List, Tuple, Optional
import threading
import faiss
import numpy as np
from .providers.embeddings import get_embedding_service
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss
from rank_bm25 import BM25Okapi


_VSTORES: Dict[str, FAISS] = {}
_BM25_INDEX: Dict[str, BM25Okapi] = {}
_DOCS: Dict[str, List[Document]] = {}

_lock = threading.RLock()
_index: Optional[faiss.Index] = None
_docs: list[dict] = []  # {"content":..., "filename":..., "metadata":...}

def init_index():
    global _index
    with _lock:
        if _index is None:
            dim = get_embedding_service().dim
            # inner product + normalized vectors => cosine
            _index = faiss.IndexFlatIP(dim)

def add_documents(docs: List[Document], collection: str) -> int:
    emb = get_embedding_service()
    # Tokenize documents for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    # If recreating or new collection, initialize new indexes
    if collection not in _VSTORES or collection not in _BM25_INDEX or collection not in _DOCS:
        # Compute embeddings for all docs (with E5 passage prefixes handled inside embed_texts)
        embeddings = emb.embed_texts([d.page_content for d in docs])
        # Create FAISS index for these embeddings
        dim = emb.dim()
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))
        docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
        index_to_id = {i: str(i) for i in range(len(docs))}
        _VSTORES[collection] = FAISS(embedding_function=emb._emb, index=index,
                                     docstore=docstore, index_to_docstore_id=index_to_id)
        _DOCS[collection] = docs
        _BM25_INDEX[collection] = BM25Okapi(tokenized_corpus)
    else:
        # Append new docs to existing indexes
        existing_store = _VSTORES[collection]
        # Compute embeddings for new docs and add to FAISS
        embeddings = emb.embed_texts([d.page_content for d in docs])
        for vec, doc in zip(embeddings, docs):
            id_str = str(len(_DOCS[collection]))
            existing_store.index.add(np.array([vec], dtype="float32"))
            existing_store.docstore.add(texts={id_str: doc})
            existing_store.index_to_docstore_id[existing_store.index.ntotal - 1] = id_str
            _DOCS[collection].append(doc)
        # Update BM25 corpus
        all_tokens = [doc.page_content.lower().split() for doc in _DOCS[collection]]
        _BM25_INDEX[collection] = BM25Okapi(all_tokens)
    return len(docs)


def search(query: str, k: int = 5) -> List[Tuple[int, dict, float]]:
    assert _index is not None, "FAISS index is not initialized. Call init_index()."
    q = get_embedding_service().encode([query], is_query=True)

    with _lock:
        D, I = _index.search(q, min(k, len(_docs) or 1))
    results: List[Tuple[int, dict, float]] = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx == -1:
            continue
        doc = _docs[idx]
        score = float(D[0][rank])
        results.append((idx, doc, score))
    return results

def get_or_create_vectorstore(collection: str, recreate: bool = False) -> FAISS:
    if recreate and collection in _VSTORES:
        # Optionally, drop existing index if recreate requested
        _VSTORES.pop(collection, None)
        _BM25_INDEX.pop(collection, None)
        _DOCS.pop(collection, None)
    if collection in _VSTORES:
        return _VSTORES[collection]
    else:
        raise VectorStoreError(f"Vector store for '{collection}' not found. Please ingest data first.")


def total_docs() -> int:
    return len(_docs)
