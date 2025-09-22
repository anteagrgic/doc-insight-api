from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    # .env i case-insensitive env lookup
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    API_TITLE: str = "Document Insight API"
    API_VERSION: str = "0.1.0"

    # --- Models (podržava i UPPERCASE ENV kao u compose-u) ---
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    qa_model: str = Field(
        default="deepset/xlm-roberta-base-squad2",
        validation_alias=AliasChoices("QA_MODEL", "qa_model"),
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("CROSS_ENCODER_MODEL", "cross_encoder_model"),
    )

    # --- Chunking ---
    CHUNK_SIZE: int = Field(
        default=800,
        validation_alias=AliasChoices("CHUNK_SIZE", "chunk_size"),
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        validation_alias=AliasChoices("CHUNK_OVERLAP", "chunk_overlap"),
    )

    # --- OCR ---
    OCR_LANGS: str = Field(
        default="en,hr",
        validation_alias=AliasChoices("OCR_LANGS", "ocr_langs"),
    )
    # dodaj u Settings:
    OCR_PDF_FALLBACK: bool = Field(default=True, env="OCR_PDF_FALLBACK")

    # --- NER ---
    ENABLE_NER: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_NER", "enable_ner"),
    )
    NER_MODEL: str = Field(
        default="en_core_web_sm",
        validation_alias=AliasChoices("NER_MODEL", "ner_model"),
    )


# instanca postavki
settings = Settings()
