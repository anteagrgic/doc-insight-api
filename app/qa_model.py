from functools import lru_cache
from transformers import pipeline
from .config import settings
from typing import List, Tuple, Optional


@lru_cache(maxsize=1)
def get_qa_pipeline():
    return pipeline("question-answering", model=settings.QA_MODEL)


def answer_from_docs(question: str, docs: List[Tuple[int, dict, float]]) -> tuple[str, Optional[int]]:
    """
    docs: lista (doc_id, doc_dict, score)
    vraća: (answer, best_doc_id)
    """
    qa = get_qa_pipeline()
    best = ("", 0.0, None)
    for doc_id, doc, _score in docs:
        ctx = doc["content"]
        if not ctx or not ctx.strip():
            continue
        res = qa(question=question, context=ctx)
        if res and res.get("score", 0.0) > best[1]:
            best = (res.get("answer", ""), float(res.get("score", 0.0)), doc_id)
    return best[0], best[2]
