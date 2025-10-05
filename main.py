from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import router
from app.providers.embeddings import get_embedding_service

def create_app() -> FastAPI:
    app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup():
        # Warmup embedding servisa (učita E5 model)
        _ = get_embedding_service()

    app.include_router(router)
    return app

app = create_app()
