"""
Aggregate every eval/out/{topic}_{variant}.csv prediction file into:
  - eval/out/summary.csv          -- one row per (topic, variant): macro P/R/F1 on
                                      the full 5-way relation label, a binary
                                      "any relation detected" P/R/F1 (fair to the
                                      keyword/embedding baselines, which were never
                                      designed to produce typed relations), API
                                      calls, and wall-clock time.
  - eval/out/confusion_{topic}_{variant}.csv  -- full 5-way confusion matrix.
  - eval/out/f1_comparison.png    -- bar chart of macro-F1 by variant, averaged
                                      across topics, for the write-up.

Usage (from inside the backend container, after run_ablation.py has produced
predictions):
    docker compose exec backend python -m eval.report
"""
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from eval.variants import NO_RELATION, VARIANTS

OUT_DIR = Path(__file__).parent / "out"

RELATION_LABELS = ["prerequisite_of", "subtopic_of", "enables", "related_to", "breaks", NO_RELATION]


def _read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score(rows: list[dict]) -> dict:
    gold = [r["gold_relation"] for r in rows]
    pred = [r["predicted_relation"] for r in rows]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=RELATION_LABELS, average="macro", zero_division=0
    )

    # Binary "any relation detected" axis -- fair to keyword/embedding baselines,
    # which can only ever say related_to or no_relation.
    gold_bin = [g != NO_RELATION for g in gold]
    pred_bin = [p != NO_RELATION for p in pred]
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        gold_bin, pred_bin, average="binary", pos_label=True, zero_division=0
    )

    api_calls = int(rows[0]["api_calls"]) if rows else 0
    wall_clock = float(rows[0]["wall_clock_seconds"]) if rows else 0.0

    return {
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "binary_precision": bin_p, "binary_recall": bin_r, "binary_f1": bin_f1,
        "api_calls": api_calls, "wall_clock_seconds": wall_clock, "n_pairs": len(rows),
    }


def _write_confusion(topic: str, variant: str, rows: list[dict]) -> None:
    gold = [r["gold_relation"] for r in rows]
    pred = [r["predicted_relation"] for r in rows]
    cm = confusion_matrix(gold, pred, labels=RELATION_LABELS)

    out_path = OUT_DIR / f"confusion_{topic}_{variant}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\predicted", *RELATION_LABELS])
        for label, row in zip(RELATION_LABELS, cm):
            writer.writerow([label, *row])


def main() -> None:
    if not OUT_DIR.exists():
        print(f"No {OUT_DIR} directory -- run run_ablation.py first.")
        return

    summary_rows = []
    f1_by_variant: dict[str, list[float]] = defaultdict(list)

    for path in sorted(OUT_DIR.glob("*_*.csv")):
        if path.name in ("summary.csv",) or path.name.startswith("confusion_"):
            continue
        # filename shape: {topic}_{variant}.csv -- variant is always one of VARIANTS
        stem = path.stem
        variant = next((v for v in VARIANTS if stem.endswith(f"_{v}")), None)
        if variant is None:
            continue
        topic = stem[: -(len(variant) + 1)]

        rows = _read_predictions(path)
        if not rows:
            continue
        scores = _score(rows)
        summary_rows.append({"topic": topic, "variant": variant, **scores})
        f1_by_variant[variant].append(scores["macro_f1"])
        _write_confusion(topic, variant, rows)

    if not summary_rows:
        print("No prediction files found in eval/out/ -- run run_ablation.py first.")
        return

    summary_path = OUT_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    for row in summary_rows:
        print(f"  {row['topic']:<20} {row['variant']:<14} "
              f"macro_f1={row['macro_f1']:.2f}  binary_f1={row['binary_f1']:.2f}  "
              f"api_calls={row['api_calls']:<4} time={row['wall_clock_seconds']:.2f}s")

    _plot_f1_comparison(f1_by_variant)


def _plot_f1_comparison(f1_by_variant: dict[str, list[float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping f1_comparison.png (see requirements.txt)")
        return

    variants_present = [v for v in VARIANTS if v in f1_by_variant]
    means = [sum(f1_by_variant[v]) / len(f1_by_variant[v]) for v in variants_present]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(variants_present, means, color=["#94a3b8", "#64748b", "#6366f1", "#22c55e"])
    ax.set_ylabel("Macro F1 (5-way relation classification)")
    ax.set_ylim(0, 1)
    ax.set_title("Relation classification: macro F1 by variant (mean across topics)")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()

    out_path = OUT_DIR / "f1_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
