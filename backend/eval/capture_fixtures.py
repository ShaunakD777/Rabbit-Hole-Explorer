"""
Freeze a source-document corpus for one seed topic into eval/fixtures/{topic}.json.

The ablation study (run_ablation.py) runs entirely offline against these frozen
fixtures rather than live connector calls, so re-running it later is deterministic
and doesn't burn free-tier LLM/API quota fetching the same sources again.

Usage (from inside the backend container, where all connector deps are installed):
    docker compose exec backend python -m eval.capture_fixtures --topic "quantum computing"
    docker compose exec backend python -m eval.capture_fixtures --topic "artificial general intelligence" --slug agi
    docker compose exec backend python -m eval.capture_fixtures --topic "blockchain"
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from app.cache import close_redis
from app.connectors.aggregator import fetch_all_sources

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _slugify(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


async def _capture(topic: str, slug: str, max_per_source: int) -> None:
    try:
        docs = await fetch_all_sources(topic, max_per_source=max_per_source)
    finally:
        # Close the shared Redis client inside this same loop, before asyncio.run()
        # tears it down -- otherwise its connection's __del__ fires after the loop
        # is already closed and logs a harmless but noisy "Event loop is closed".
        await close_redis()
    payload = [
        {
            "source_type": d.source_type,
            "url": d.url,
            "title": d.title,
            "raw_text": d.raw_text,
            "author_or_channel": d.author_or_channel,
            "published_at": d.published_at.isoformat() if d.published_at else None,
        }
        for d in docs
    ]
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / f"{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} documents to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Seed topic query, e.g. 'quantum computing'")
    parser.add_argument("--slug", default=None, help="Fixture filename stem (default: slugified topic)")
    parser.add_argument("--max-per-source", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_capture(args.topic, args.slug or _slugify(args.topic), args.max_per_source))
