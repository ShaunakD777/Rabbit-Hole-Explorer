# Ablation study: relation-classification variants

This harness compares four ways of typing the edges between a fixed set of
concepts, per the project's own implementation plan (`06_Implementation_Plan...md`,
Phase 7):

| Variant | What it does | API calls per graph |
|---|---|---|
| `keyword` | co-occurrence in the same ~300-word chunk of source text | 0 |
| `embedding` | cosine similarity of local sentence embeddings | 0 |
| `llm_pairwise` | the legacy one-call-per-pair classifier | O(n² − n) |
| `llm_batched` | the current default: one call classifies the whole graph | 1 |

`keyword` and `embedding` can only ever say `related_to` or nothing (they have no
concept of *type* of relation) — `report.py` scores them on both the full 5-way
label and a binary "any relation detected" axis so they aren't penalized purely for
a distinction they were never designed to make.

## Running it

From the host, against the running `backend` container:

```bash
# 1. Freeze a source corpus for a topic (only needs to be done once per topic)
docker compose exec backend python -m eval.capture_fixtures --topic "quantum computing" --slug quantum_computing

# 2. Run all four variants against that topic's gold labels
docker compose exec backend python -m eval.run_ablation --topic quantum_computing --variant all

# 3. Aggregate results: eval/out/summary.csv, eval/out/confusion_*.csv, eval/out/f1_comparison.png
docker compose exec backend python -m eval.report
```

`llm_pairwise` and `llm_batched` call whichever free-tier provider is configured in
`.env` (see `app/nlp/llm.py`) — if none is configured, both fall back to
`related_to` at confidence 0.3 for every pair (the same "LLM unavailable" fallback
`classify_relation()` already has), which will visibly tank their scores. Configure
at least one provider before running the LLM variants for a meaningful comparison.

## About `labels/quantum_computing.csv`

This is a **14-pair starter set** I hand-labeled from the actual captured corpus, to
prove the harness runs end-to-end and produces sane numbers on every metric
(including at least one `no_relation` negative pair, so that class isn't
untested). It is **not** the research-grade dataset the implementation plan calls
for.

The plan's own Phase 7 specifies **150–300 hand-labeled pairs across 2–3 seed
topics** (it names Quantum Computing, AGI, and Blockchain) as the empirical basis
for the paper's results section — and explicitly frames labeling as a human task,
not something to automate. To extend this:

1. `capture_fixtures.py` for each additional seed topic (`--slug agi`, `--slug blockchain`, ...).
2. Pick a concept set per topic (e.g. run extraction once, or choose canonical
   concepts by hand) and label pairs as `prerequisite_of` / `subtopic_of` /
   `enables` / `related_to` / `breaks` / `no_relation` in a new
   `labels/{slug}.csv`, matching this file's two-column-plus-label shape.
2. Re-run `run_ablation.py --topic {slug} --variant all` for each, then `report.py`
   once at the end -- it aggregates every `eval/out/*.csv` file it finds.

The plan's own contingency (§4) allows shrinking to two seed topics if labeling
150–300 pairs runs long -- that's still presentable, not a corner cut.

## Ambiguity-check eval

`nlp/intent.py::check_ambiguity()` decides whether a topic query names one
coherent subject or several mutually exclusive subjects (e.g. "Football" ->
American football vs. soccer vs. rugby, not just classic cross-domain homonyms
like "Mercury" -> planet vs. element vs. god). `labels/ambiguity.csv` is a
20-query hand-labeled set (10 ambiguous, 10 not) covering both cross-domain
homonyms and same-category-but-mutually-exclusive cases, used to check that a
wording change to `AMBIGUITY_PROMPT` actually improves detection instead of
just fixing the one query it was eyeballed against.

```bash
docker compose exec backend python -m eval.run_ambiguity_eval
```

Prints a per-query OK/MISS line, writes `eval/out/ambiguity_predictions.csv`,
and reports accuracy/precision/recall/F1. Each run deletes the relevant cache
entries first, since the cache key is the query text alone and would otherwise
mask a prompt change behind a stale cached answer. Needs at least one LLM
provider configured -- with none configured, `check_ambiguity()` always
returns "not ambiguous", which would score every ambiguous-gold row as a miss.
