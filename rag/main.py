import uvicorn
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.v1.router import router
from src.settings.llamaindex_settings import initialize_settings
from src.settings.config import RagSettings
from src.services.chroma_manager import ChromaManager

settings = RagSettings()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.async_client = httpx.AsyncClient(timeout=60.0, verify=False)
    app.state.sync_client = httpx.Client(timeout=60.0, verify=False)
    
    await initialize_settings(
        aclient=app.state.async_client,
        client=app.state.sync_client
    )
    
    # Initialize ChromaManager singleton
    chroma_manager = ChromaManager()
    chroma_manager.initialize()
    
    yield
    await app.state.async_client.aclose()
    app.state.sync_client.close()
    

app = FastAPI(
    title="RAG",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)
app.include_router(router, prefix="/api/v1", tags=["RAG"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=int(settings.port),
        timeout_keep_alive=300,
        limit_concurrency=1000
    )