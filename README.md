# Document Insight API — Hybrid RAG with OCR (FAISS+BM25 · E5 · CrossEncoder · XLM‑R QA · spaCy NER)

This project implements a **hybrid retrieval‑augmented question answering** pipeline over your PDFs/images/text using:

- **Embeddings:** `intfloat/multilingual-e5-base` (with proper `query:` / `passage:` prefixes)
- **Dense retrieval:** **FAISS** (inner product on L2‑normalized vectors)
- **Sparse retrieval:** **BM25** (`rank-bm25`)
- **Score fusion:** normalized 50/50 dense + sparse
- **Reranking:** **CrossEncoder** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Extractive QA:** **XLM‑R** `deepset/xlm-roberta-base-squad2`
- **NER (optional):** **spaCy** `en_core_web_sm`
- **OCR fallback for scanned PDFs/images:** **EasyOCR** (with tunable `max_pages` and `dpi`)

Everything runs **in‑memory** (no Qdrant). Models are **preloaded in the Dockera image** for fast startup.


## What’s included

- `app/api.py` — FastAPI router with `/health`, `/upload`, `/ask`
- `app/embeddings.py` — E5 embedding service
- `app/ingest.py` — PDF parsing, OCR, helpers
- `app/models.py` — Pydantic request/response models
- `app/config.py` — Settings via environment (compose-friendly)
- `scripts/preload_models.py` — Pre-download all HF/EasyOCR models during image build
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- Tests: `tests/` (smoke, TOC fallback, scanned‑PDF ingest, etc.)


## API Endpoints

### `GET /health`
Returns basic status and model names.

```json
{
  "status": "ok",
  "docs_indexed": 42,
  "faiss_index": 42,
  "bm25": true,
  "ner": true,
  "embedding_model": "intfloat/multilingual-e5-base",
  "qa_model": "deepset/xlm-roberta-base-squad2",
  "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### `POST /upload`
Multipart file ingest. Supports **PDF**, **PNG/JPG/TIFF**, **plain text**.  
For **PDFs** we parse page‑by‑page. If `page.get_text()` returns empty (scanned pages), we **OCR** the page image.

**Query params (optional):**
- `max_pages` — limit pages per PDF for speed (default: unlimited; example: `15`)
- `dpi` — rasterization DPI for OCR pages (default: 120–150 reasonable)

**Response:**
```json
{ "num_files": 2, "chunks_indexed": 47 }
```

**Metadata captured per chunk:**  
`metadata = { "page": <int>, "source": <filename>, "ocr": <bool> }`


### `POST /ask`
Body:
```json
{
  "question": "Who was the first person to walk on the Moon?",
  "k": 5,
  "mode": "fusion"   // one of: "fusion" | "embedding" | "bm25"
}
```

Response:
```json
{
  "answer": "Neil Armstrong",
  "sources": [
    {
      "id": 1,
      "filename": "note.txt",
      "content": "Neil Armstrong was the first person to walk on the Moon.",
      "score": 11.36,
      "metadata": { "page": 1, "source": "note.txt", "ocr": false }
    }
  ],
  "entities": [
    { "text": "Neil Armstrong", "label": "PERSON" }
  ]
}
```

> **Note:** If the QA model doesn’t find an answer in the top reranked passage (or confidence is too low), you’ll get `"answer": "No answer found."` and the top‑k reranked `sources`. `entities` will be `null` when NER is disabled or when there’s no answer text to analyze.


## How retrieval works (pipeline)

1. **Ingest**: files → pages (PDF) / OCR (images or scanned pages) → chunks (here each page is treated as one chunk) → store text + metadata.
2. **Index**:
   - **FAISS** over **E5** passage vectors (normalized; inner‑product = cosine).
   - **BM25** over tokenized page text.
3. **Query**:
   - Embed query as E5 `query:`.
   - Retrieve top‑N from FAISS and BM25.
   - **Score fusion** (normalize each list; 50/50 sum) and dedupe by `(source, page)`.
   - **Rerank** fused candidates with **CrossEncoder** (top‑5).
   - **QA** with XLM‑R over the **best candidate**; fallback returns top sources.
   - Optional **NER** over `answer` (if enabled).


## Docker & Compose

### 0) Clean out old stack (optional but recommended)
```bash
docker compose down -v
docker image prune -f
docker builder prune -f
```

### 1) Build (models preloaded in image)
```bash
docker compose build --no-cache
```

### 2) Run
```bash
docker compose up -d
docker compose logs -f
# visit: http://localhost:8000/health
```

The Compose mounts `./models` into the container to **persist HF/EasyOCR cache** between runs.


## Quick cURL demo

### Health
```bash
curl -s http://localhost:8000/health | jq
```

### Upload simple text
```bash
curl -s -X POST http://localhost:8000/upload \
  -F 'files=@./test_files/note.txt;type=text/plain' | jq
```

### Upload scanned PDF with limits (faster)
```bash
curl -s -X POST 'http://localhost:8000/upload?max_pages=15&dpi=120' \
  -F 'files=@./test_files/MTP_materijali_s_predavanja.pdf;type=application/pdf' | jq
```

### Ask (fusion = FAISS+BM25)
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Koji su glavni sadržaji skripte?","k":5,"mode":"fusion"}' | jq
```

### Ask (dense only)
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who was the first person to walk on the Moon?","k":5,"mode":"embedding"}' | jq
```


## Running tests

Inside the running container:
```bash
docker compose exec app pytest -q
```

Typical tests included:
- `test_api_smoke.py` — health → upload → ask happy path
- `test_toc_fallback.py` — TOC‑like Croatian query and metadata checks
- `test_ingest_scanned_pdf.py` — ensures OCR path is hit and indexed when PDF text is empty


## Configuration (env vars)

Set via `docker-compose.yml` and read in `app/config.py`:

- `EMBEDDING_MODEL` (default: `intfloat/multilingual-e5-base`)
- `QA_MODEL` (default: `deepset/xlm-roberta-base-squad2`)
- `CROSS_ENCODER_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `ENABLE_NER` (`true`/`false` — default `false`)
- `NER_MODEL` (default: `en_core_web_sm`)
- `OCR_LANGS` (default: `en,hr`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP` (if you later switch to finer chunking)
- HF caches: `HF_HOME`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`

> **Note:** `entities` field in `/ask` is `null` when `ENABLE_NER=false` or when no answer text is produced.


## Performance tips

- **Scanned PDFs**: Use `max_pages` and a moderate `dpi` (e.g., 120–150) on `/upload` to keep OCR fast.
- **Reranking**: We rerank only the top‑5 fused candidates to keep latency down.
- **Inner-product on normalized vectors** ≈ cosine similarity, which works well with FAISS `IndexFlatIP`.
- For big corpora, consider chunking pages and increasing FAISS `k` (and BM25 top‑N) judiciously.


## Troubleshooting

- **`/health` shows `docs_indexed: 0`** — you haven’t called `/upload` successfully.
- **OCR slow / request times out** — limit `max_pages` and lower `dpi` (120–150).
- **`entities: null`** — enable NER (`ENABLE_NER=true`) and ensure the QA produced non‑empty `answer`.
- **FAISS count not increasing** — confirm `/upload` returns `chunks_indexed > 0` and that you haven’t hit a parse/OCR failure.
- **Model download errors** — models are preloaded by `scripts/preload_models.py` during build; ensure the image was rebuilt after changes (`--no-cache`).


## Git LFS (optional for big PDFs)

```bash
# install once on your machine (macOS example)
brew install git-lfs
git lfs install

# track large PDFs
git lfs track "test_files/*.pdf"
git add .gitattributes

# if .gitignore excludes test_files, override with -f
git add -f test_files/*.pdf
git commit -m "Add sample PDFs via Git LFS"
git push
```

If you **don’t** want large binaries in the repo, don’t add the PDFs; keep them locally and only use `/upload` at runtime.


## Security & Privacy

- All indexing happens **in‑memory** unless you mount persistent volumes.
- Uploaded content is only kept inside the container and the mapped `./models` cache (for model weights), not your file data.


## License

MIT (or your choice). Update as appropriate.
