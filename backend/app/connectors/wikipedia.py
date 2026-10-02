"""Wikipedia connector using the MediaWiki REST API."""
import logging
import httpx
from app.connectors.base import BaseConnector, RawDocument
from app.cache import cache_get, cache_set, source_key
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

WIKI_API = "https://en.wikipedia.org/api/rest_v1"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia rejects requests with no User-Agent, or a generic default one such as
# "python-httpx/x.y" (what httpx sends if you don't set one), with a 403 -- see
# https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
# Every request this connector made was being blocked outright before it ever got
# a chance to run.
_HEADERS = {"User-Agent": "RabbitHoleExplorer/1.0 (academic NLP course project)"}


class WikipediaConnector(BaseConnector):
    source_type = "wikipedia"

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        cache_k = source_key("wikipedia", query)
        cached = await cache_get(cache_k)
        if cached:
            logger.debug("Wikipedia cache hit for '%s'", query)
            return [RawDocument.from_cached_dict(d) for d in cached]

        results = await self._search_and_fetch(query, max_results)
        await cache_set(cache_k, [self._to_dict(r) for r in results],
                        ttl=settings.ttl_source_cache)
        return results

    async def _search_and_fetch(self, query: str, max_results: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
            # 1. Search for page titles
            search_resp = await client.get(WIKI_SEARCH_API, params={
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "format": "json",
            })
            search_resp.raise_for_status()
            _, titles, _, urls = search_resp.json()

            for title, url in zip(titles[:max_results], urls[:max_results]):
                try:
                    # 2. Fetch full page summary
                    encoded = title.replace(" ", "_")
                    resp = await client.get(f"{WIKI_API}/page/summary/{encoded}")
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    extract = data.get("extract", "")
                    if not extract:
                        continue

                    # 3. Fetch the full plain-text extract (all sections, not just
                    # the summary) via the classic action API. The REST
                    # "mobile-sections" endpoint this used to call has been fully
                    # decommissioned (403 "Mobile Content Service is
                    # decommissioned"), which was silently degrading every
                    # article down to just its one-paragraph summary.
                    full_text = extract
                    content_resp = await client.get(WIKI_SEARCH_API, params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "titles": title,
                        "format": "json",
                    })
                    if content_resp.status_code == 200:
                        pages = content_resp.json().get("query", {}).get("pages", {})
                        page = next(iter(pages.values()), {})
                        full_extract = page.get("extract", "")
                        if full_extract:
                            full_text = full_extract

                    docs.append(RawDocument(
                        source_type="wikipedia",
                        url=url,
                        title=title,
                        raw_text=full_text[:8000],  # cap to avoid huge texts
                        metadata={"page_id": data.get("pageid")},
                    ))
                except Exception as exc:
                    logger.warning("Wikipedia fetch failed for '%s': %s", title, exc)

        logger.info("Wikipedia returned %d docs for '%s'", len(docs), query)
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
