from functools import lru_cache
from typing import List
from .config import settings
from .models import Entity

def ner_enabled() -> bool:
    return bool(settings.ENABLE_NER)

@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load(settings.NER_MODEL)
    except Exception as e:
        # ako model nije dostupan, fallback (NER off)
        return None

def extract_entities(text: str) -> List[Entity]:
    if not ner_enabled():
        return []
    nlp = _nlp()
    if nlp is None or not text:
        return []
    doc = nlp(text)
    ents: List[Entity] = []
    for e in doc.ents:
        ents.append(Entity(text=e.text, label=e.label_))
    return ents
