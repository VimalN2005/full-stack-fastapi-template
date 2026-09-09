from fastapi import APIRouter

from app.api.routes import ai, chat, items, login, private, rag, users, utils
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(items.router)
api_router.include_router(rag.router)
api_router.include_router(ai.router)
api_router.include_router(chat.router)


if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)
