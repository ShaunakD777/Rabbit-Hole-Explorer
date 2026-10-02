from fastapi import APIRouter
from app.api.v1 import topics, nodes, search, progress, trends, videos

router = APIRouter()
router.include_router(topics.router, prefix="/topics", tags=["topics"])
router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
router.include_router(search.router, prefix="/graph", tags=["search"])
router.include_router(progress.router, prefix="/progress", tags=["progress"])
router.include_router(trends.router, prefix="/trends", tags=["trends"])
router.include_router(videos.router, prefix="/videos", tags=["videos"])
