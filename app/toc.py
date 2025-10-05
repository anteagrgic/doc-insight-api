import re

TOC_TRIGGERS = ["sadržaj", "contents", "table of contents", "sadrzaj"]

def looks_like_toc_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in TOC_TRIGGERS)

def extract_toc_block(text: str) -> str | None:
    # hvata blok od 'SADRŽAJ' do sljedeće velike sekcije ili dvije prazne linije
    m = re.search(r"(SADRŽAJ.*?)(?:\n{2,}|(?=\n\s*\d+\.\s*[A-ZŠĐČĆŽ]))", text, flags=re.S | re.I)
    if m:
        # očisti tab-točke i višak razmaka
        block = re.sub(r"[·\.]{2,}", " … ", m.group(1))
        block = re.sub(r"[ \t]+", " ", block)
        return block.strip()
    return None
