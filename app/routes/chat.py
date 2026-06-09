"""
Chat router — exposes two endpoints:

  POST /api/v1/chat/query   → full JSON answer
  POST /api/v1/chat/stream  → Server-Sent Events stream
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import QueryRequest, QueryResponse
from app.services import qa_service, transcript_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _get_retriever(request: QueryRequest, settings: Settings):
    """Build (or return a cached) vector-store retriever for the video."""
    try:
        return transcript_service.get_or_build_retriever(
            video_id=request.video_id,
            language=request.language,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error building retriever: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the YouTube transcript.",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question about a YouTube video (blocking)",
)
async def query(request: QueryRequest, settings: SettingsDep) -> QueryResponse:
    """
    Retrieve the transcript for *video_id*, run RAG, and return the full
    answer in a single JSON response.
    """
    retriever = _get_retriever(request, settings)
    docs = await retriever.ainvoke(request.query)

    try:
        answer = await qa_service.get_answer(
            query=request.query,
            docs=docs,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("QA inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Inference request to HuggingFace failed.",
        ) from exc

    return QueryResponse(
        video_id=request.video_id,
        query=request.query,
        answer=answer,
    )


@router.post(
    "/stream",
    summary="Answer a question about a YouTube video (streaming SSE)",
)
async def stream(request: QueryRequest, settings: SettingsDep) -> StreamingResponse:
    """
    Same as `/query` but returns tokens incrementally via Server-Sent Events.
    Suitable for real-time display in the Chrome extension popup.
    """
    retriever = _get_retriever(request, settings)
    docs = await retriever.ainvoke(request.query)

    async def event_generator():
        try:
            async for token in qa_service.stream_answer(
                query=request.query,
                docs=docs,
                settings=settings,
            ):
                # SSE format: "data: <payload>\n\n"
                yield f"data: {token}\n\n"
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield "event: error\ndata: Inference failed\n\n"
        finally:
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering if proxied
        },
    )
