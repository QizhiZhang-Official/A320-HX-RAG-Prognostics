from fastapi import FastAPI
from pathlib import Path

from fastapi.responses import FileResponse
from ragnostics.api.dependencies import lifespan
from ragnostics.api.routes.health import router as health_router
from ragnostics.api.routes.retrieve import router as retrieve_router

FRONTEND_INDEX = Path(__file__).resolve().parents[3] / 'frontend' / 'index.html'


app = FastAPI(title="A320-HX-RAG-API", version='0.1.0', lifespan=lifespan)
app.include_router(router=health_router, prefix='/api')
app.include_router(router=retrieve_router, prefix='/api')

@app.get('/', include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)