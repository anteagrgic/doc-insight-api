from __future__ import annotations
from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import settings


class EmbeddingService:
    """
    Jedinstveni servis za embedinge.
    - Podržava E5 prefikse ("query:" / "passage:")
    - Normalizira vektore (kosinus)
    - Izlaže i dim property i dim() metodu (radi kompatibilnosti)
    """
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or getattr(settings, "embedding_model", "intfloat/multilingual-e5-base")
        self.model = SentenceTransformer(self.model_name)
        self._is_e5 = "e5" in self.model_name.lower()
        # Odredi dimenziju:
        probe = self.model.encode(["probe"], normalize_embeddings=True)
        self._dim = int(probe.shape[1])

    # Property pristup
    @property
    def dim(self) -> int:
        return self._dim

    # I opcionalno .dim() za stariji kod
    def dim_(self) -> int:
        return self._dim

    # Neki dijelovi koda zovu emb.dim() – podrži i to.
    def dim__(self) -> int:
        return self._dim

    # Back-compat: ako netko zove emb.dim(), preusmjeri na property
    def __call_dim_fallback(self):
        return self._dim

    # Glavna metoda
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype="float32")

        # E5 prefiksi
        if self._is_e5:
            if is_query:
                texts = [t if t.startswith("query:") else f"query: {t}" for t in texts]
            else:
                texts = [t if t.startswith("passage:") else f"passage: {t}" for t in texts]

        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # bitno za kosinus/IP u FAISS-u
        )
        return vecs.astype("float32")

    # Alias koji neki tvoji moduli očekuju
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts, is_query=False).tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    # Prioritetno koristi settings.embedding_model (lowercase)
    model_name = getattr(settings, "embedding_model", None)
    if not model_name:
        # Fallback ako je netko postavio ENV EMBEDDING_MODEL, a nije mapirano
        model_name = getattr(settings, "EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return EmbeddingService(model_name)


# --- Helperi praktični za korištenje u drugim modulima ---

def embed_query(text: str) -> np.ndarray:
    """
    Vrati 1D vektor upita (E5 'query:' + normalizacija).
    """
    svc = get_embedding_service()
    v = svc.encode([text], is_query=True)
    return v[0]

def embed_passages(passages: List[str]) -> np.ndarray:
    """
    Vrati 2D matricu vektora za dokumente (E5 'passage:' + normalizacija).
    """
    svc = get_embedding_service()
    return svc.encode(passages, is_query=False)
