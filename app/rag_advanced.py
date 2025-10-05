# At the top of the file, import additional requirements
from sentence_transformers import CrossEncoder
from transformers import pipeline
import spacy

# Load spaCy model once (for NER)
_nlp = spacy.load("en_core_web_sm")

def answer_advanced(question: str, *, k: int = 5, translate: bool | None = None,
                    mode: str = "simple", n_queries: int = 4, k_smooth: int = 60) -> AdvancedRAGResult:
    t0 = perf_counter()
    vs = get_or_create_vectorstore(getattr(settings, "qdrant_collection", "default"))
    # (Optional) translation step (omitted for brevity; assume query is in English or model is multilingual)
    q = question  # use original question since embedding model is multilingual E5

    # 2) Retrieval
    docs = []
    if mode == "fusion":
        # **Hybrid retrieval:** use both vector search and BM25
        emb = get_embedding_service()
        # Embed query with E5 (with "query:" prefix)
        query_vec = emb.embed_texts([f"query: {q}"])[0]
        # Dense retrieval (FAISS) – get top 10 similar docs
        results = vs.similarity_search_with_score_by_vector(query_vec, k=10)
        dense_docs = [doc for doc, score in results]
        dense_scores = [score for doc, score in results]  # L2 distances (lower is better)
        # BM25 retrieval – get top 10 docs by BM25 score
        bm25 = _BM25_INDEX[getattr(settings, "qdrant_collection", "default")]
        tokenized_query = q.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)  # scores for all docs
        # Get top 10 BM25 docs and scores
        all_docs = _DOCS[getattr(settings, "qdrant_collection", "default")]
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_docs = [all_docs[i] for i in top_bm25_indices]
        bm25_top_scores = [bm25_scores[i] for i in top_bm25_indices]

        # Normalize scores (min-max) for fusion
        if dense_scores:
            ds_min, ds_max = min(dense_scores), max(dense_scores)
            dense_norm = [(ds_max - s) / (ds_max - ds_min + 1e-9) for s in dense_scores]  # invert distance
        else:
            dense_norm = []
        if bm25_top_scores:
            bm25_min, bm25_max = min(bm25_top_scores), max(bm25_top_scores)
            bm25_norm = [(s - bm25_min) / (bm25_max - bm25_min + 1e-9) for s in bm25_top_scores]
        else:
            bm25_norm = []

        # Merge results by combined score
        combined_candidates: Dict[str, Tuple[Document, float]] = {}
        for doc, score_norm in zip(dense_docs, dense_norm):
            doc_id = doc.metadata.get("doc_id", None)
            combined_candidates[doc_id] = (doc, score_norm)  # use doc_id as key
        for doc, score_norm in zip(bm25_docs, bm25_norm):
            doc_id = doc.metadata.get("doc_id", None)
            if doc_id in combined_candidates:
                # If also in dense, add scores
                prev_doc, prev_score = combined_candidates[doc_id]
                combined_candidates[doc_id] = (prev_doc, prev_score + score_norm)
            else:
                combined_candidates[doc_id] = (doc, score_norm)
        # Sort by combined score (descending) and take top 20
        combined_list = sorted(combined_candidates.values(), key=lambda x: x[1], reverse=True)
        top_candidates = combined_list[:20]  # (Document, combined_score)

        # **Cross-encoder reranking** on top 20:
        reranker = CrossEncoder(settings.cross_encoder_model)
        pairs = [(q, doc.page_content) for doc, _ in top_candidates]
        cross_scores = reranker.predict(pairs)  # higher = more relevant
        # Sort the candidates by cross-encoder score
        ranked = sorted(zip(top_candidates, cross_scores), key=lambda x: x[1], reverse=True)
        # Take top-k documents for final answer context
        docs = [doc for (doc, _), score in ranked[:k]]
        # Optionally, store the cross-encoder score in metadata for output
        for (doc, combined_score), score in ranked[:k]:
            doc.metadata["score"] = float(score)
    else:
        # **Simple retrieval:** vector search only (FAISS)
        emb = get_embedding_service()
        query_vec = emb.embed_texts([f"query: {q}"])[0]
        results = vs.similarity_search_with_score_by_vector(query_vec, k=k)
        docs = [doc for doc, score in results]
        # Store similarity score in metadata if desired
        for doc, score in results:
            doc.metadata["score"] = float(score)
    retrieval_time = perf_counter() - t0

    # 3) Extractive QA: use the QA model to find answer from retrieved docs
    qa_pipe = pipeline("question-answering", model=settings.qa_model, tokenizer=settings.qa_model, handle_impossible_answer=True)
    answer_text = ""
    used_doc = None
    for doc in docs:
        context_text = doc.page_content
        result = qa_pipe(question=q, context=context_text)
        ans = result.get("answer", "").strip()
        if ans:
            answer_text = ans
            used_doc = doc
            break
    if not answer_text:
        # No answer found in any top doc -> fallback
        answer_text = "No answer found."

    # 4) Named Entity Recognition (NER) on the answer
    doc_nlp = _nlp(answer_text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc_nlp.ents]

    # Prepare response
    sourcedocs = as_sourcedocs(docs[:k])
    return AdvancedRAGResult(
        answer=answer_text,
        sources=[SourceDoc(**sd.dict()) for sd in sourcedocs],
        meta={"retrieval_time": f"{retrieval_time:.3f}s", "used_document": used_doc.metadata.get("doc_id") if used_doc else None,
              "mode": mode, "translated": False if translate is None else bool(translate)}
    )
