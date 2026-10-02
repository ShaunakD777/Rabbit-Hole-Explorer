"""Papers connector: Semantic Scholar API with arXiv fallback."""
import asyncio
import logging
from datetime import datetime, timezone

import httpx

from app.connectors.base import BaseConnector, RawDocument
from app.cache import cache_get, cache_set, source_key
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"


def _el_text(el) -> str:
    """Text of an XML Element, or "" -- always compare Elements to None, never
    truthiness (an Element with no *child elements* is falsy regardless of text)."""
    return el.text or "" if el is not None else ""


class PapersConnector(BaseConnector):
    source_type = "paper"

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        cache_k = source_key("paper", query)
        cached = await cache_get(cache_k)
        if cached:
            logger.debug("Papers cache hit for '%s'", query)
            return [RawDocument.from_cached_dict(d) for d in cached]

        results = await self._fetch_semantic_scholar(query, max_results)
        if not results:
            results = await self._fetch_arxiv(query, max_results)

        await cache_set(cache_k, [self._to_dict(r) for r in results],
                        ttl=settings.ttl_source_cache)
        return results

    async def _fetch_semantic_scholar(self, query: str, max_results: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(SS_SEARCH_URL, params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,abstract,authors,year,externalIds,url",
                }, headers=headers)
                resp.raise_for_status()
                for paper in resp.json().get("data", []):
                    abstract = paper.get("abstract") or ""
                    title = paper.get("title", "")
                    authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
                    year = paper.get("year")
                    pub_at = datetime(year, 1, 1, tzinfo=timezone.utc) if year else None
                    url = paper.get("url") or ""

                    docs.append(RawDocument(
                        source_type="paper",
                        url=url,
                        title=title,
                        raw_text=f"{title}\n\nAuthors: {authors}\n\n{abstract}",
                        author_or_channel=authors,
                        published_at=pub_at,
                        metadata={
                            "paper_id": paper.get("paperId"),
                            "year": year,
                        },
                    ))
        except Exception as exc:
            logger.warning("Semantic Scholar fetch failed for '%s': %s", query, exc)
        return docs

    async def _fetch_arxiv(self, query: str, max_results: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(ARXIV_SEARCH_URL, params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                })
                resp.raise_for_status()

                import xml.etree.ElementTree as ET
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)
                for entry in root.findall("atom:entry", ns):
                    # `entry.find(...) or {}` looks like a safe None-fallback, but
                    # ElementTree's Element defines __len__ (child element count) and
                    # no __bool__, so any leaf element with text but no *child
                    # elements* -- title, summary, published, every author's name --
                    # is falsy even though find() found it. `or {}` then always
                    # kicked in and {}.text raised AttributeError on every entry.
                    title = _el_text(entry.find("atom:title", ns))
                    summary = _el_text(entry.find("atom:summary", ns))
                    link_el = entry.find("atom:link[@rel='alternate']", ns)
                    url = link_el.attrib.get("href", "") if link_el is not None else ""
                    published_str = _el_text(entry.find("atom:published", ns))
                    published_at = (
                        datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        if published_str else None
                    )
                    authors = ", ".join(
                        _el_text(a.find("atom:name", ns))
                        for a in entry.findall("atom:author", ns)[:3]
                    )
                    docs.append(RawDocument(
                        source_type="paper",
                        url=url,
                        title=title.strip(),
                        raw_text=f"{title.strip()}\n\nAuthors: {authors}\n\n{summary.strip()}",
                        author_or_channel=authors,
                        published_at=published_at,
                        metadata={"source": "arxiv"},
                    ))
        except Exception as exc:
            logger.warning("arXiv fetch failed for '%s': %s", query, exc)
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
