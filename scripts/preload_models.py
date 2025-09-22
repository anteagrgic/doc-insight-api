# /app/scripts/preload_models.py
import os

def preload_hf_models():
    # E5 embedder
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("[preload] Downloading E5 embedder...")
    SentenceTransformer("intfloat/multilingual-e5-base")

    # Cross-encoder reranker
    print("[preload] Downloading CrossEncoder reranker...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # XLM-R QA (tokenizer + model)
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    print("[preload] Downloading XLM-R QA tokenizer/model...")
    AutoTokenizer.from_pretrained("deepset/xlm-roberta-base-squad2")
    AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-base-squad2")


def preload_easyocr():
    # EasyOCR preuzima modele on-demand; ovdje barem iniciramo Reader da povuče jezike
    try:
        import easyocr
        langs = os.getenv("OCR_LANGS", "en,hr")
        lang_list = [x.strip() for x in langs.split(",") if x.strip()]
        print(f"[preload] Initializing EasyOCR for languages: {lang_list}")
        _ = easyocr.Reader(lang_list, gpu=False)  # ovo će povući potrebne modele u /models/easyocr
        print("[preload] EasyOCR models prepared.")
    except Exception as e:
        print(f"[preload] EasyOCR preload skipped or failed: {e}")


if __name__ == "__main__":
    print("[preload] Starting model downloads into HF/EasyOCR caches...")
    preload_hf_models()
    preload_easyocr()
    print("[preload] Done.")
