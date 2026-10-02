"""POST /videos/{source_document_id}/extract — YouTube extraction."""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import SourceDocument, VideoExtraction
from app.nlp.llm import complete_json, AllProvidersFailed

logger = logging.getLogger(__name__)
router = APIRouter()

NOTES_PROMPT = """\
Given this YouTube video transcript, produce:
1. A structured notes outline with 4–6 bullet points (markdown).
2. A list of 5 key concepts (short labels, comma-separated).
3. 3–5 multiple-choice quiz questions.

Respond with JSON only:
{{
  "notes_outline": "markdown string",
  "key_concepts": ["concept1", "concept2", ...],
  "quiz": [
    {{"question": "...", "options": ["A", "B", "C", "D"], "correct_index": 0}},
    ...
  ]
}}

Transcript (first 4000 chars):
{transcript}
"""


@router.post("/{source_document_id}/extract")
async def extract_video(
    source_document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Trigger (or return cached) YouTube extraction for a source document."""
    # Check cache
    stmt = select(VideoExtraction).where(
        VideoExtraction.source_document_id == source_document_id
    )
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        return {
            "notes_outline": existing.notes_outline,
            "key_concepts": existing.key_concepts,
            "quiz": existing.quiz,
        }

    # Load source document
    doc_stmt = select(SourceDocument).where(SourceDocument.id == source_document_id)
    doc = (await db.execute(doc_stmt)).scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Source document not found")
    if doc.source_type != "youtube":
        raise HTTPException(status_code=400, detail="Source is not a YouTube document")
    if not doc.raw_text:
        raise HTTPException(status_code=422, detail="No transcript available for this video")

    # Generate with the free-tier LLM provider chain (app/nlp/llm.py)
    prompt = NOTES_PROMPT.format(transcript=doc.raw_text[:4000])

    try:
        data = await complete_json(prompt, max_tokens=2048)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got: {type(data)}")
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=503, detail=f"No LLM provider configured/available: {exc}")
    except Exception as exc:
        logger.exception("Video extraction LLM call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    extraction = VideoExtraction(
        source_document_id=source_document_id,
        notes_outline=data.get("notes_outline", ""),
        key_concepts=data.get("key_concepts", []),
        quiz=data.get("quiz", []),
    )
    db.add(extraction)
    await db.commit()

    return {
        "notes_outline": extraction.notes_outline,
        "key_concepts": extraction.key_concepts,
        "quiz": extraction.quiz,
    }
