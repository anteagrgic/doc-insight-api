from __future__ import annotations

from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field


# -------------------- Request modeli --------------------

class AskRequest(BaseModel):
    """
    Ulaz za /ask endpoint.
    - question: upit korisnika (obavezno)
    - k: broj izvora koje vraćamo (default 5)
    - mode: način dohvata ('fusion' = FAISS+BM25, 'embedding' = samo FAISS, 'bm25' = samo BM25)
    """
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    mode: Optional[Literal["fusion", "embedding", "bm25"]] = Field(
        default="fusion",
        description="Retrieval mode: 'fusion' (FAISS+BM25), 'embedding' (FAISS only), or 'bm25' (BM25 only).",
    )


# -------------------- Entity / Meta modeli --------------------

class Entity(BaseModel):
    text: str
    label: str


class MetaInfo(BaseModel):
    """
    Neobavezne meta-informacije o izvršenju upita (ako ih želiš vraćati).
    Nije nužno za /ask, ali QueryResponse ga referencira pa ga definiramo.
    """
    mode: Optional[str] = None                  # npr. 'fusion'
    retrieval_time_sec: Optional[float] = None  # ukupno vrijeme dohvata (ako mjeriš)
    used_document_page: Optional[int] = None    # stranica iz koje je QA izvukao odgovor
    used_document_source: Optional[str] = None  # filename izvora


# -------------------- Source / Response modeli --------------------

class SourceDoc(BaseModel):
    id: int
    filename: Optional[str] = None
    content: str
    score: Optional[float] = None
    # OVDJE ostavljamo sve dodatne informacije (uklj. 'source' i 'page')
    metadata: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    # Ako je NER isključen u settings, ovo će biti None (što je ok)
    entities: Optional[List[Entity]] = None


# Ovaj model nije nužan za /ask, ali ga ostavljamo jer si ga ranije imala.
# Ako ga ne koristiš, možeš ga i ukloniti iz koda.
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    meta: Optional[MetaInfo] = None
    entities: Optional[List[Entity]] = None


class UploadResponse(BaseModel):
    num_files: int
    chunks_indexed: int
