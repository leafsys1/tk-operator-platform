from fastapi import APIRouter

from ..config import settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"ok": True, "app": settings.app_name, "db": settings.db_url}
