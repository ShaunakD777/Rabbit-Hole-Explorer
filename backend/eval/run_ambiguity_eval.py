"""
Measure nlp/intent.py::check_ambiguity() against a hand-labeled set of queries,
so a prompt change (wording, examples, criteria) can be checked against a fixed
set of known-ambiguous and known-unambiguous topics instead of a few queries
tried by hand.

Requires eval/labels/ambiguity.csv (query,gold_ambiguous,notes) and at least one
LLM provider configured in .env -- check_ambiguity() degrades to "not ambiguous"
with no provider configured, which would make every ambiguous-gold row a miss
and silently produce a meaningless 50% score.

Always deletes each query's ambiguity cache entry before calling check_ambiguity()
-- the cache key is keyed on the query text alone, not on the prompt's wording, so
a cached answer from before a prompt change would otherwise mask the very thing
this harness exists to measure.

Usage (from inside the backend container):
    docker compose exec backend python -m eval.run_ambiguity_eval
"""
import asyncio
import csv
from pathlib import Path

from app.cache import cache_delete, close_redis, llm_ambiguity_key
from app.nlp.intent import check_ambiguity

BASE_DIR = Path(__file__).parent
LABELS_PATH = BASE_DIR / "labels" / "ambiguity.csv"
OUT_PATH = BASE_DIR / "out" / "ambiguity_predictions.csv"


def _load_labels() -> list[tuple[str, bool, str]]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"No gold labels at {LABELS_PATH}.")
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["query"], row["gold_ambiguous"].strip().lower() == "true", row.get("notes", ""))
            for row in reader
        ]


async def main() -> None:
    gold = _load_labels()

    rows = []
    tp = fp = tn = fn = 0
    try:
        for query, gold_ambiguous, notes in gold:
            await cache_delete(llm_ambiguity_key(query))
            candidates = await check_ambiguity(query)
            predicted_ambiguous = candidates is not None
            correct = predicted_ambiguous == gold_ambiguous

            if predicted_ambiguous and gold_ambiguous:
                tp += 1
            elif predicted_ambiguous and not gold_ambiguous:
                fp += 1
            elif not predicted_ambiguous and not gold_ambiguous:
                tn += 1
            else:
                fn += 1

            senses = "; ".join(c["label"] for c in candidates) if candidates else ""
            rows.append([query, gold_ambiguous, predicted_ambiguous, correct, senses, notes])
            marker = "OK" if correct else "MISS"
            print(f"[{marker}] {query!r}: gold={gold_ambiguous} predicted={predicted_ambiguous}"
                  + (f" -> {senses}" if senses else ""))
    finally:
        await close_redis()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "gold_ambiguous", "predicted_ambiguous", "correct",
                          "predicted_senses", "notes"])
        writer.writerows(rows)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{total} queries -- accuracy={accuracy:.2f}  "
          f"precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Predictions written to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
