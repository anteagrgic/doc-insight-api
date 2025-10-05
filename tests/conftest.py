import pytest

from app.api import DOCUMENTS, DOC_METAS, FAISS_INDEX, BM25_INDEX

@pytest.fixture(autouse=True)
def reset_rag_state():
    """
    Automatski se izvršava prije SVAKOG testa.
    Resetira globalno stanje u app.api tako da testovi budu izolirani.
    """
    # očisti postojeće
    DOCUMENTS.clear()
    DOC_METAS.clear()

    # resetiraj indekse
    from app import api as api_module
    api_module.FAISS_INDEX = None
    api_module.BM25_INDEX = None

    yield

    # (nije nužno ništa nakon testa)
