"""
Run one or all relation-classification variants against a topic's frozen fixtures
and gold labels, writing per-pair predictions to eval/out/{topic}_{variant}.csv.

Usage (from inside the backend container):
    docker compose exec backend python -m eval.run_ablation --topic quantum_computing --variant all
    docker compose exec backend python -m eval.run_ablation --topic quantum_computing --variant llm_batched

Requires eval/fixtures/{topic}.json (capture_fixtures.py) and
eval/labels/{topic}.csv (hand-labeled: concept_a,concept_b,gold_relation).
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path

from eval.variants import (
    run_keyword, run_embedding, run_llm_pairwise, run_llm_batched, NO_RELATION, VARIANTS,
)

BASE_DIR = Path(__file__).parent
FIXTURES_DIR = BASE_DIR / "fixtures"
LABELS_DIR = BASE_DIR / "labels"
OUT_DIR = BASE_DIR / "out"


def _load_labels(topic: str) -> list[tuple[str, str, str]]:
    path = LABELS_DIR / f"{topic}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No gold labels at {path}. Hand-label pairs as "
            f"'concept_a,concept_b,gold_relation' -- see labels/README.md."
        )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["concept_a"], row["concept_b"], row["gold_relation"]) for row in reader]


def _load_corpus_text(topic: str) -> str:
    path = FIXTURES_DIR / f"{topic}.json"
    if not path.exists():
        raise FileNotFoundError(f"No fixtures at {path}. Run capture_fixtures.py first.")
    docs = json.loads(path.read_text(encoding="utf-8"))
    return " ".join((d.get("raw_text") or "")[:2000] for d in docs[:8])


async def run_variant(name: str, topic: str, pairs, concepts, corpus_text):
    if name == "keyword":
        return run_keyword(pairs, corpus_text)
    if name == "embedding":
        return run_embedding(pairs)
    if name == "llm_pairwise":
        return await run_llm_pairwise(pairs, corpus_text[:1500])
    if name == "llm_batched":
        return await run_llm_batched(pairs, concepts, corpus_text[:1500])
    raise ValueError(f"Unknown variant: {name}")


def _write_predictions(topic: str, variant: str, gold, result) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{topic}_{variant}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_a", "concept_b", "gold_relation", "predicted_relation",
                          "api_calls", "wall_clock_seconds"])
        for a, b, gold_relation in gold:
            predicted = result.predictions.get((a, b), NO_RELATION)
            writer.writerow([a, b, gold_relation, predicted,
                              result.api_calls, f"{result.wall_clock_seconds:.3f}"])
    return out_path


async def main(topic: str, variant: str) -> None:
    gold = _load_labels(topic)
    pairs = [(a, b) for a, b, _ in gold]
    concepts = list(dict.fromkeys(c for pair in pairs for c in pair))  # unique, order-preserving
    corpus_text = _load_corpus_text(topic)

    names = VARIANTS if variant == "all" else [variant]
    for name in names:
        result = await run_variant(name, topic, pairs, concepts, corpus_text)
        out_path = _write_predictions(topic, name, gold, result)
        print(f"[{name}] {len(pairs)} pairs, {result.api_calls} API call(s), "
              f"{result.wall_clock_seconds:.2f}s -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Fixture/label slug, e.g. 'quantum_computing'")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()
    asyncio.run(main(args.topic, args.variant))
