"""YouTube connector: search + transcript extraction."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from app.connectors.base import BaseConnector, RawDocument
from app.cache import cache_get, cache_set, source_key
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeConnector(BaseConnector):
    source_type = "youtube"

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        if not settings.youtube_api_key:
            logger.warning("YouTube API key not configured — skipping")
            return []

        cache_k = source_key("youtube", query)
        cached = await cache_get(cache_k)
        if cached:
            logger.debug("YouTube cache hit for '%s'", query)
            return [RawDocument.from_cached_dict(d) for d in cached]

        results = await self._search_and_fetch(query, max_results)
        await cache_set(cache_k, [self._to_dict(r) for r in results],
                        ttl=settings.ttl_source_cache)
        return results

    async def _search_and_fetch(self, query: str, max_results: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        async with httpx.AsyncClient(timeout=15) as client:
            # Search for videos
            search_resp = await client.get(YT_SEARCH_URL, params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "relevanceLanguage": "en",
                "key": settings.youtube_api_key,
            })
            search_resp.raise_for_status()
            items = search_resp.json().get("items", [])

            video_ids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
            if not video_ids:
                return docs

            # Get video details (duration, view count etc.)
            detail_resp = await client.get(YT_VIDEO_URL, params={
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids),
                "key": settings.youtube_api_key,
            })
            detail_resp.raise_for_status()
            details = {v["id"]: v for v in detail_resp.json().get("items", [])}

        for vid_id in video_ids:
            try:
                detail = details.get(vid_id, {})
                snippet = detail.get("snippet", {})
                stats = detail.get("statistics", {})

                title = snippet.get("title", "")
                channel = snippet.get("channelTitle", "")
                published_raw = snippet.get("publishedAt")
                published_at = (
                    datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    if published_raw else None
                )

                # youtube-transcript-api is synchronous/blocking; run it off the
                # event loop so one slow transcript fetch can't stall every other
                # coroutine sharing this loop (previously called directly here).
                transcript_text = await asyncio.get_running_loop().run_in_executor(
                    None, self._get_transcript, vid_id
                )

                docs.append(RawDocument(
                    source_type="youtube",
                    url=f"https://www.youtube.com/watch?v={vid_id}",
                    title=title,
                    raw_text=transcript_text or f"{title}\n{snippet.get('description', '')}",
                    author_or_channel=channel,
                    published_at=published_at,
                    metadata={
                        "video_id": vid_id,
                        "view_count": stats.get("viewCount"),
                        "has_transcript": bool(transcript_text),
                    },
                ))
            except Exception as exc:
                logger.warning("YouTube fetch failed for video %s: %s", vid_id, exc)

        logger.info("YouTube returned %d docs for '%s'", len(docs), query)
        return docs

    @staticmethod
    def _get_transcript(video_id: str) -> Optional[str]:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return " ".join(entry["text"] for entry in transcript)
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as exc:
            logger.debug("Transcript fetch failed for %s: %s", video_id, exc)
            return None

    @staticmethod
    def _to_dict(doc: RawDocument) -> dict:
        return {
            "source_type": doc.source_type,
            "url": doc.url,
            "title": doc.title,
            "raw_text": doc.raw_text,
            "author_or_channel": doc.author_or_channel,
            "published_at": doc.published_at.isoformat() if doc.published_at else None,
            "metadata": doc.metadata,
        }
