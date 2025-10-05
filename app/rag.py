from typing import List
from .vectorstore import search
from .qa_model import answer_from_docs
from .models import SourceDoc
from .toc import looks_like_toc_question, extract_toc_block


def ask(question: str, k: int = 5) -> tuple[str, List[SourceDoc]]:
    hits = search(question, k=k)
    if not hits:
        return "Nije pronađen relevantan kontekst.", []
   
    # TOC fallback: prođi kroz top hitove i pokušaj izvući blok
    if looks_like_toc_question(question):
        for _, doc, _ in hits[:min(5, len(hits))]:
            toc = extract_toc_block(doc["content"])
            if toc and len(toc.split()) > 6:
                src = [SourceDoc(id=_, filename=doc.get("filename"), content=doc["content"], score=_, metadata=doc.get("metadata") or {}) for _, doc, _ in hits[:1]]
                return toc, src

    #inace normalan QA
    answer, best_id = answer_from_docs(question, hits)
    sources: List[SourceDoc] = []
    for idx, doc, score in hits:
        sd = SourceDoc(id=idx, filename=doc.get("filename"), content=doc["content"], score=score, metadata=doc.get("metadata") or {})
        sources.append(sd)
    # Ako želiš vratiti samo izvor iz kojeg je došao odgovor:
    if best_id is not None:
        sources = [s for s in sources if s.id == best_id] or sources[:1]
    return answer or "Nije pronađen jasan odgovor u kontekstu.", sources
