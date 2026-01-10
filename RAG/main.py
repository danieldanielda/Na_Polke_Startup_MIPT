import uvicorn
import httpx
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.v1.router import router
from src.settings.llamaindex_settings import initialize_settings
from src.settings.config import RagSettings

settings = RagSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.async_client = httpx.AsyncClient(timeout=60.0, verify=False)
    app.state.sync_client = httpx.Client(timeout=60.0, verify=False)
    
    await initialize_settings(
        aclient=app.state.async_client,
        client=app.state.sync_client
    )
    yield
    await app.state.async_client.aclose()
    app.state.sync_client.close()
    

app = FastAPI(
    title="RAG",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)
app.include_router(router, prefix="/api/v1/rag", tags=["RAG"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host, 
        port=int(settings.port),
        timeout_keep_alive=30,
        limit_concurrency=1000,
        log_level="info"
    )