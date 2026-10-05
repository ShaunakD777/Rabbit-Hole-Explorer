"""
Tests for connectors/aggregator.py::build_combined_context -- the LLM context
sampler. Pure logic over unsaved SourceDocument objects; no DB or network.
"""
from app.connectors.aggregator import build_combined_context, MIN_CONTEXT_DOC_CHARS
from app.db.models import SourceDocument


def _doc(source_type: str, marker: str, length: int = 1000) -> SourceDocument:
    return SourceDocument(source_type=source_type, url=f"https://x/{marker}",
                          raw_text=(marker + " ") * (length // (len(marker) + 1)))


def test_papers_take_their_turn_after_introductory_sources():
    docs = [_doc("paper", "PAPER"), _doc("wikipedia", "WIKI"), _doc("youtube", "VIDEO")]

    context = build_combined_context(docs, per_doc_chars=100, max_total_chars=10_000)

    assert context.index("WIKI") < context.index("VIDEO") < context.index("PAPER")


def test_wikipedia_turn_is_larger_than_other_sources():
    docs = [_doc("wikipedia", "WIKI", 5000), _doc("paper", "PAPER", 5000)]

    context = build_combined_context(docs, per_doc_chars=100, max_total_chars=10_000)

    assert context.count("WIKI") > 2 * context.count("PAPER")


def test_snippet_sized_docs_skipped_when_longer_material_exists():
    docs = [_doc("web", "SNIPPET", MIN_CONTEXT_DOC_CHARS - 50), _doc("wikipedia", "WIKI")]

    context = build_combined_context(docs, per_doc_chars=800, max_total_chars=10_000)

    assert "SNIPPET" not in context and "WIKI" in context


def test_snippets_still_used_when_they_are_all_there_is():
    docs = [_doc("web", "SNIPPET", 150)]
    assert "SNIPPET" in build_combined_context(docs)
