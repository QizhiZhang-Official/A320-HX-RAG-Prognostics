from fastapi import FastAPI

from ragnostics.api.dependencies import lifespan
from ragnostics.api.routes.health import router as health_router
from ragnostics.api.routes.retrieve import router as retrieve_router


app = FastAPI(title="A320-HX-RAG-API", version='0.1.0', lifespan=lifespan)
app.include_router(router=health_router, prefix='/api')
app.include_router(router=retrieve_router, prefix='/api')