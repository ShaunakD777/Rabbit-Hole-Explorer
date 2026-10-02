"""Reddit connector using PRAW (sync) wrapped in a thread executor."""
import asyncio
import logging
from datetime import datetime, timezone

import praw

from app.connectors.base import BaseConnector, RawDocument
from app.cache import cache_get, cache_set, source_key
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedditConnector(BaseConnector):
    source_type = "reddit"

    def _build_client(self) -> praw.Reddit:
        return praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
            check_for_async=False,
        )

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        if not settings.reddit_client_id:
            logger.warning("Reddit credentials not configured — skipping")
            return []

        cache_k = source_key("reddit", query)
        cached = await cache_get(cache_k)
        if cached:
            logger.debug("Reddit cache hit for '%s'", query)
            return [RawDocument.from_cached_dict(d) for d in cached]

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, self._sync_fetch, query, max_results
        )
        await cache_set(cache_k, [self._to_dict(r) for r in results],
                        ttl=settings.ttl_source_cache)
        return results

    def _sync_fetch(self, query: str, max_results: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        try:
            reddit = self._build_client()
            submissions = reddit.subreddit("all").search(
                query, sort="relevance", time_filter="year", limit=max_results
            )
            for sub in submissions:
                # Compile top-level comments + post body
                post_text = f"{sub.title}\n\n{sub.selftext}"
                sub.comments.replace_more(limit=0)
                top_comments = [c.body for c in sub.comments.list()[:10] if hasattr(c, "body")]
                full_text = post_text + "\n\n" + "\n".join(top_comments)

                created = datetime.fromtimestamp(sub.created_utc, tz=timezone.utc)
                docs.append(RawDocument(
                    source_type="reddit",
                    url=f"https://reddit.com{sub.permalink}",
                    title=sub.title,
                    raw_text=full_text[:6000],
                    author_or_channel=str(sub.author) if sub.author else None,
                    published_at=created,
                    metadata={
                        "subreddit": sub.subreddit.display_name,
                        "score": sub.score,
                        "num_comments": sub.num_comments,
                    },
                ))
        except Exception as exc:
            logger.warning("Reddit fetch failed for '%s': %s", query, exc)
        logger.info("Reddit returned %d docs for '%s'", len(docs), query)
        return docs

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
