from fastapi import APIRouter
from ragnostics.api.schemas import HealthResponse


router = APIRouter(prefix='/health', tags=['health'])

@router.get("")
def health() -> HealthResponse:
    return HealthResponse(status='ok')