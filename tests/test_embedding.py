import numpy as np
from app.providers.embeddings import get_embedding_service

def test_embedding_dim_and_norm():
    svc = get_embedding_service()
    vecs = svc.encode(["hello world", "hello world"])
    assert vecs.shape[1] == svc.dim
    # normalizirano (kosinus 1.0 isti tekst)
    cos = float(np.dot(vecs[0], vecs[1]))
    assert cos > 0.99
