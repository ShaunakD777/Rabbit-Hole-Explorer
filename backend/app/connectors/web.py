"""General web/blog/news search connector via SerpAPI, falling back to Serper."""
import logging
import httpx

from app.connectors.base import BaseConnector, RawDocument
from app.cache import cache_get, cache_set, source_key
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SERPAPI_URL = "https://serpapi.com/search"
SERPER_URL = "https://google.serper.dev/search"


def _keys_for(raw: str) -> list[str]:
    """
    A *_api_key setting may hold one key or several comma-separated keys --
    multiple free-tier accounts for the same provider. Tried in order so one
    account's exhausted quota falls through to the next instead of failing the
    whole search (see llm.py's identically-named helper for the LLM chain).
    """
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


class WebConnector(BaseConnector):
    source_type = "web"

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        serpapi_keys = _keys_for(settings.serpapi_key)
        serper_keys = _keys_for(settings.serper_api_key)
        if not serpapi_keys and not serper_keys:
            logger.warning("No SerpAPI/Serper key configured -- skipping web search")
            return []

        cache_k = source_key("web", query)
        cached = await cache_get(cache_k)
        if cached:
            logger.debug("Web search cache hit for '%s'", query)
            return [RawDocument.from_cached_dict(d) for d in cached]

        results = await self._search(query, max_results, serpapi_keys, serper_keys)
        await cache_set(cache_k, [self._to_dict(r) for r in results],
                        ttl=settings.ttl_source_cache)
        return results

    async def _search(
        self, query: str, max_results: int, serpapi_keys: list[str], serper_keys: list[str]
    ) -> list[RawDocument]:
        async with httpx.AsyncClient(timeout=15) as client:
            for api_key in serpapi_keys:
                docs = await self._search_serpapi(client, query, max_results, api_key)
                if docs:
                    return docs
            for api_key in serper_keys:
                docs = await self._search_serper(client, query, max_results, api_key)
                if docs:
                    return docs
        logger.warning("Web search returned no docs for '%s' (all keys/providers exhausted)", query)
        return []

    async def _search_serpapi(
        self, client: httpx.AsyncClient, query: str, max_results: int, api_key: str
    ) -> list[RawDocument]:
        try:
            resp = await client.get(SERPAPI_URL, params={
                "q": query,
                "api_key": api_key,
                "num": max_results,
                "hl": "en",
            })
            resp.raise_for_status()
            organic = resp.json().get("organic_results", [])
            docs = [
                RawDocument(
                    source_type="web",
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    raw_text=f"{item.get('title', '')}\n\n{item.get('snippet', '')}",
                    metadata={"source": "serpapi"},
                )
                for item in organic[:max_results]
            ]
            logger.info("SerpAPI search returned %d docs for '%s'", len(docs), query)
            return docs
        except Exception as exc:
            logger.warning("SerpAPI search failed for '%s': %s", query, exc)
            return []

    async def _search_serper(
        self, client: httpx.AsyncClient, query: str, max_results: int, api_key: str
    ) -> list[RawDocument]:
        try:
            resp = await client.post(
                SERPER_URL,
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            organic = resp.json().get("organic", [])
            docs = [
                RawDocument(
                    source_type="web",
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    raw_text=f"{item.get('title', '')}\n\n{item.get('snippet', '')}",
                    metadata={"source": "serper"},
                )
                for item in organic[:max_results]
            ]
            logger.info("Serper search returned %d docs for '%s'", len(docs), query)
            return docs
        except Exception as exc:
            logger.warning("Serper search failed for '%s': %s", query, exc)
            return []

    @staticmethod
    def _to_dict(doc: RawDocument) -> dict:
        return {
            "source_type": doc.source_type,
            "url": doc.url,
            "title": doc.title,
            "raw_text": doc.raw_text,
            "author_or_channel": doc.author_or_channel,
            "published_at": None,
            "metadata": doc.metadata,
        }
