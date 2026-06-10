"""
Chat router — exposes three endpoints:

  GET  /api/v1/chat/languages → available transcript languages for a video
  POST /api/v1/chat/query     → full JSON answer
  POST /api/v1/chat/stream    → Server-Sent Events stream
"""

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import LanguagesResponse, QueryRequest, QueryResponse, TranscriptLanguage
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
@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="List available transcript languages for a YouTube video",
)
async def languages(
    settings: SettingsDep,
    video_id: str = Query(
        ...,
        min_length=1,
        max_length=20,
        description="YouTube video ID (e.g. 'hVIfPT0RWlo')",
        examples=["hVIfPT0RWlo"],
    ),
) -> LanguagesResponse:
    """
    Returns every transcript language available for the given video.
    The first entry in the list is the recommended default.
    If no transcripts exist the response will be a 422 with an explanation.
    """
    try:
        langs = transcript_service.get_available_languages(video_id, settings)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error fetching languages for '%s': %s", video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch transcript languages.",
        ) from exc

    if not langs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No transcripts found for video '{video_id}'.",
        )

    return LanguagesResponse(
        video_id=video_id,
        languages=[TranscriptLanguage(**lang) for lang in langs],
    )


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
    Streams tokens via Server-Sent Events.

    Before the first token arrives the client receives ``event: status``
    messages describing each build stage so the UI can show live progress:

    .. code-block:: text

        event: status
        data: Fetching transcript from YouTube...

        event: status
        data: Split into 42 chunks. Generating embeddings...

        data: Here           # answer tokens start
        data:  is
        data:  your answer.

        event: done
        data: [DONE]
    """

    async def event_generator():
        loop = asyncio.get_running_loop()
        # Queue bridges the sync worker thread → async generator
        progress_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def on_progress(msg: str) -> None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, msg)

        async def _build():
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: transcript_service.get_or_build_retriever(
                        video_id=request.video_id,
                        language=request.language,
                        settings=settings,
                        on_progress=on_progress,
                    ),
                )
            finally:
                # Always unblock the queue consumer, even on error
                progress_queue.put_nowait(None)

        retriever_task = asyncio.create_task(_build())

        # Relay progress messages until the sentinel (None) arrives
        while True:
            msg = await progress_queue.get()
            if msg is None:
                break
            yield f"event: status\ndata: {msg}\n\n"

        try:
            retriever = await retriever_task
        except ValueError as exc:
            yield f"event: error\ndata: {exc}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        except Exception as exc:
            logger.exception("Unexpected error building retriever: %s", exc)
            yield "event: error\ndata: Failed to process the YouTube transcript.\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        yield "event: status\ndata: Thinking...\n\n"
        docs = await retriever.ainvoke(request.query)

        try:
            async for token in qa_service.stream_answer(
                query=request.query,
                docs=docs,
                settings=settings,
            ):
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
