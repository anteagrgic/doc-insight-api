FROM python:3.11-slim

# --- OS deps (PyMuPDF ok na slim; EasyOCR/opencv trebaju libgl & glib) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- HF/EasyOCR cache (persistirat ćemo ih kroz volume u compose) ---
ENV HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf \
    EASYOCR_MODULE_PATH=/models/easyocr \
    PYTHONUNBUFFERED=1

# --- Kopiraj requirements unaprijed radi layer cache-a ---
COPY requirements.txt /app/requirements.txt

# --- Instaliraj Python ovisnosti (CPU wheel index već u requirements.txt) ---
RUN pip install --default-timeout=1200 --no-cache-dir -r /app/requirements.txt

# --- spaCy model (NER) ---
# umjesto: RUN python -m spacy download en_core_web_sm
RUN pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl" \
 && python -m spacy validate


# --- Kopiraj aplikaciju ---
COPY app /app/app
COPY main.py /app/main.py
COPY scripts /app/scripts

# --- Preload modela (HF, EasyOCR) – čita OCR_LANGS iz build ARG-a ili ENV ---
ARG OCR_LANGS="en,hr"
ENV OCR_LANGS=${OCR_LANGS}

# U ovoj skripti eksplicitno povlačimo:
# - E5 embedder
# - CrossEncoder reranker
# - XLM-R QA (tokenizer + model)
# - EasyOCR jezike (koliko je moguće)
RUN python /app/scripts/preload_models.py

EXPOSE 8000

# --- Start ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
