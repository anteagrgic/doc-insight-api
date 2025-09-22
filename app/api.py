from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import settings
from .models import AskRequest, AskResponse, UploadResponse, SourceDoc, Entity
from .ingest import parse_image, parse_text  # OCR/tekst koristimo i dalje
from .providers.embeddings import embed_query, embed_passages

# ----------------- Router -----------------
router = APIRouter()

# ----------------- Global state -----------------
DOCUMENTS: List[str] = []              # svaki element je jedan chunk/stranica
DOC_METAS: List[Dict[str, Any]] = []   # paralelno s DOCUMENTS (npr. {"source": fn, "page": 3})
FAISS_INDEX: Optional[faiss.Index] = None
BM25_INDEX: Optional[BM25Okapi] = None

# Lazy global modeli
QA_PIPELINE = None
NER_NLP = None
CROSS_ENCODER = None  # reranker


# ----------------- Helpers -----------------
def _ensure_reranker():
    global CROSS_ENCODER
    if CROSS_ENCODER is None:
        ce_name = getattr(settings, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        CROSS_ENCODER = CrossEncoder(ce_name)

def _sourcedoc(idx: int, text: str, meta: Dict[str, Any], score: float) -> SourceDoc:
    return SourceDoc(
        id=idx,
        filename=meta.get("source"),
        content=text,
        score=float(score),
        metadata=dict(meta) if meta else {},
    )


# ----------------- Routes -----------------
@router.get("/health")
def health():
    return {
        "status": "ok",
        "docs_indexed": len(DOCUMENTS),
        "faiss_index": int(FAISS_INDEX.ntotal) if FAISS_INDEX is not None else 0,
        "bm25": bool(BM25_INDEX is not None),
        "ner": bool(getattr(settings, "ENABLE_NER", False)),
        "embedding_model": getattr(settings, "embedding_model", None),
        "qa_model": getattr(settings, "qa_model", "deepset/xlm-roberta-base-squad2"),
        "cross_encoder_model": getattr(settings, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    }


@router.post("/upload", response_model=UploadResponse)
async def upload(
    files: List[UploadFile] = File(...),
    max_pages: Optional[int] = Query(
        None, ge=1,
        description="Obradi najviše ovoliko stranica po PDF-u (OCR je spor, korisno za brzi test)."
    ),
    dpi: int = Query(
        150, ge=72, le=300,
        description="Rasterizacija PDF stranica za OCR fallback (viši DPI = bolje ali sporije)."
    ),
):
    """
    Ingest: PDF (po stranicama), Images (OCR), Text. Index: FAISS (dense) + BM25 (sparse).

    - PDF: pokušaj `page.get_text('text')`; ako prazno -> rasteriziraj na zadani DPI i pokreni OCR (EasyOCR).
    - Slike: direktno OCR.
    - Tekst: direktni decode.
    """
    if not files:
        raise HTTPException(400, "No files provided.")

    import fitz  # PyMuPDF

    new_texts: List[str] = []
    new_metas: List[Dict[str, Any]] = []
    indexed_filenames: List[str] = []

    # DPI -> scale (PDF je 72 dpi bazno)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    for f in files:
        data = await f.read()
        if not data:
            continue
        lower = f.filename.lower()

        if lower.endswith(".pdf"):
            try:
                doc = fitz.open(stream=data, filetype="pdf")
            except Exception:
                # loš PDF, preskoči
                continue
            any_added = False
            for page_index, page in enumerate(doc):
                if max_pages and page_index >= max_pages:
                    break

                # 1) Pokušaj nativni tekst
                text = (page.get_text("text") or "").strip()
                meta = {"page": page_index + 1, "source": f.filename}

                # 2) Ako nema teksta -> OCR fallback (rasteriziraj stranicu pa parse_image)
                if not text:
                    try:
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img_bytes = pix.tobytes("png")
                        text = (parse_image(img_bytes) or "").strip()
                        if text:
                            meta["ocr"] = True  # označi da je došlo OCR-om
                    except Exception:
                        text = ""

                if not text:
                    continue  # prazna stranica i nakon fallbacka

                new_texts.append(text)
                new_metas.append(meta)
                any_added = True

            if any_added:
                indexed_filenames.append(f.filename)

        elif lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
            # OCR za slike
            t = (parse_image(data) or "").strip()
            if t:
                new_texts.append(t)
                new_metas.append({"source": f.filename, "page": 1, "ocr": True})
                indexed_filenames.append(f.filename)

        else:
            # običan tekst
            t = (parse_text(data) or "").strip()
            if t:
                new_texts.append(t)
                new_metas.append({"source": f.filename, "page": 1})
                indexed_filenames.append(f.filename)

    if not new_texts:
        raise HTTPException(400, "No parsable content.")

    # Append u in-memory docstore
    DOCUMENTS.extend(new_texts)
    DOC_METAS.extend(new_metas)

    # ---- BM25: build/update ----
    global BM25_INDEX
    tokenized_docs = [d.lower().split() for d in DOCUMENTS]
    BM25_INDEX = BM25Okapi(tokenized_docs)

    # ---- FAISS: build/update (E5 "passage:" + normalizacija) ----
    global FAISS_INDEX
    to_embed = new_texts if FAISS_INDEX is not None else DOCUMENTS
    emb = embed_passages(to_embed).astype("float32")
    if FAISS_INDEX is None:
        FAISS_INDEX = faiss.IndexFlatIP(emb.shape[1])  # inner-product (s unit vektorima = cosine)
        FAISS_INDEX.add(emb)
    else:
        FAISS_INDEX.add(emb)

    return UploadResponse(
        num_files=len(indexed_filenames),
        chunks_indexed=len(new_texts),
    )


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """
    Hybrid retrieval: FAISS + BM25 -> score fusion -> CrossEncoder rerank -> QA (XLM-R SQuAD2) -> spaCy NER
    - 'mode': 'fusion' (oba), 'embedding' (samo FAISS), 'bm25' (samo BM25)
    """
    question = (req.question or "").strip()
    mode = (req.mode or "fusion").lower()
    if mode not in ("fusion", "embedding", "bm25"):
        mode = "fusion"

    if not question:
        raise HTTPException(400, "Question is empty.")
    if not DOCUMENTS:
        raise HTTPException(400, "No documents available for search.")

    # --- Retrieval prema odabranom mode ---
    faiss_results: List[Dict[str, Any]] = []
    bm25_results: List[Dict[str, Any]] = []

    if mode in ("fusion", "embedding"):
        # Dense (FAISS)
        qvec = embed_query(question)
        D, I = FAISS_INDEX.search(np.array([qvec], dtype="float32"), k=min(10, len(DOCUMENTS)))
        for sc, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            faiss_results.append({"text": DOCUMENTS[idx], "meta": DOC_METAS[idx], "score": float(sc)})

    if mode in ("fusion", "bm25"):
        # Sparse (BM25)
        scores = BM25_INDEX.get_scores(question.lower().split())
        top = np.argsort(scores)[-10:][::-1]
        for idx in top:
            sc = float(scores[idx])
            if sc <= 0:
                continue
            bm25_results.append({"text": DOCUMENTS[idx], "meta": DOC_METAS[idx], "score": sc})

    # --- Kombinacija / fusion ---
    combined: Dict[Tuple[Optional[str], Optional[int]], Dict[str, Any]] = {}

    if mode == "embedding":
        for r in faiss_results:
            k = (r["meta"].get("source"), r["meta"].get("page"))
            combined[k] = {"text": r["text"], "meta": r["meta"], "score": r["score"]}
    elif mode == "bm25":
        for r in bm25_results:
            k = (r["meta"].get("source"), r["meta"].get("page"))
            combined[k] = {"text": r["text"], "meta": r["meta"], "score": r["score"]}
    else:  # fusion -> 50/50 normalizirani zbroj
        max_f = max((r["score"] for r in faiss_results), default=0.0) or 1.0
        max_b = max((r["score"] for r in bm25_results), default=0.0) or 1.0
        for r in faiss_results:
            k = (r["meta"].get("source"), r["meta"].get("page"))
            combined[k] = {"text": r["text"], "meta": r["meta"], "score": 0.5 * (r["score"] / max_f)}
        for r in bm25_results:
            k = (r["meta"].get("source"), r["meta"].get("page"))
            add = 0.5 * (r["score"] / max_b)
            if k in combined:
                combined[k]["score"] += add
            else:
                combined[k] = {"text": r["text"], "meta": r["meta"], "score": add}

    candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    if not candidates:
        return AskResponse(answer="No answer found.", sources=[], entities=None)

    # --- CrossEncoder reranking (top-5) ---
    _ensure_reranker()
    top_candidates = candidates[:5]
    pairs = [(question, c["text"]) for c in top_candidates]
    scores_ce = CROSS_ENCODER.predict(pairs)  # higher = more relevant
    for c, s in zip(top_candidates, scores_ce):
        c["rerank_score"] = float(s)
    top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Najbolji kandidat
    best_doc = top_candidates[0]
    best_text = best_doc["text"]
    best_meta = best_doc["meta"]

    # -------- QA PIPELINE --------
    global QA_PIPELINE
    if 'QA_PIPELINE' not in globals() or QA_PIPELINE is None:
        from transformers import pipeline
        QA_PIPELINE = pipeline(
            "question-answering",
            model="deepset/xlm-roberta-base-squad2",
            tokenizer="deepset/xlm-roberta-base-squad2",
            handle_impossible_answer=True
        )

    qa_input = {"question": question, "context": best_text}
    result = QA_PIPELINE(qa_input)
    answer_text = (result.get("answer", "") or "").strip()
    answer_score = float(result.get("score", 0.0))

    if answer_text == "" or answer_score < 1e-4:
        # Fallback: vrati top-5 izvora
        sources = [
            _sourcedoc(idx=i + 1,
                       text=c["text"],
                       meta=c["meta"],
                       score=float(c.get("rerank_score", c.get("score", 0.0))))
            for i, c in enumerate(top_candidates)
        ]
        return AskResponse(
            answer="No answer found.",
            sources=sources,
            entities=None
        )

    # -------- NER (opcionalno) --------
    ents_payload: List[Entity] = []
    if bool(getattr(settings, "ENABLE_NER", False)):
        global NER_NLP
        if 'NER_NLP' not in globals() or NER_NLP is None:
            import spacy
            NER_NLP = spacy.load(getattr(settings, "NER_MODEL", "en_core_web_sm"))
        doc = NER_NLP(answer_text)
        for ent in doc.ents:
            ents_payload.append(Entity(text=ent.text, label=ent.label_))

    # Izvori (top-5)
    sources = [
        _sourcedoc(idx=i + 1,
                   text=c["text"],
                   meta=c["meta"],
                   score=float(c.get("rerank_score", c.get("score", 0.0))))
        for i, c in enumerate(top_candidates)
    ]

    return AskResponse(
        answer=answer_text,
        sources=sources,
        entities=(ents_payload or None)
    )
