"""
Transcript service — fetches, splits, embeds, and caches a YouTube transcript
into a per-video-id Chroma vector store so that repeated queries on the same
video do NOT re-embed the transcript.
"""

import logging
from threading import Lock
from typing import Dict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from app.config import Settings

logger = logging.getLogger(__name__)

# In-process cache: video_id → Chroma retriever
_store_cache: Dict[str, Chroma] = {}
_cache_lock = Lock()


def _build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Construct the HuggingFace embedding model (CPU-friendly defaults)."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _fetch_transcript(video_id: str, language: str) -> str:
    """
    Fetch the YouTube transcript for the given video_id.

    Falls back to English if the requested language is unavailable.

    Raises:
        ValueError: If the transcript cannot be fetched for any reason.
    """
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language])
    except NoTranscriptFound:
        logger.warning(
            "No '%s' transcript for %s — attempting English fallback.",
            language,
            video_id,
        )
        try:
            transcript = ytt_api.fetch(video_id, languages=["en"])
        except NoTranscriptFound as exc:
            raise ValueError(
                f"No transcript found for video '{video_id}' in '{language}' or 'en'."
            ) from exc
    except TranscriptsDisabled as exc:
        raise ValueError(
            f"Transcripts are disabled for video '{video_id}'."
        ) from exc
    except VideoUnavailable as exc:
        raise ValueError(f"Video '{video_id}' is unavailable.") from exc

    return " ".join(snippet.text for snippet in transcript)


def get_or_build_retriever(video_id: str, language: str, settings: Settings):
    """
    Return a cached LangChain retriever for the given video_id.

    If no vector store exists yet it will be built from the live transcript.
    Thread-safe via a module-level lock.
    """
    cache_key = f"{video_id}:{language}"

    # Fast path — already cached
    with _cache_lock:
        if cache_key in _store_cache:
            logger.info("Cache hit for video '%s'.", video_id)
            return _store_cache[cache_key].as_retriever(
                search_type="mmr",
                search_kwargs={"k": settings.retriever_k},
            )

    # Slow path — fetch, split, embed, then cache
    logger.info("Building vector store for video '%s'.", video_id)
    full_text = _fetch_transcript(video_id, language)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.create_documents([full_text])

    if not chunks:
        raise ValueError(f"Transcript for video '{video_id}' produced no text chunks.")

    embeddings = _build_embeddings(settings)
    vector_store = Chroma.from_documents(chunks, embeddings)

    with _cache_lock:
        _store_cache[cache_key] = vector_store

    logger.info(
        "Vector store for '%s' built with %d chunks.", video_id, len(chunks)
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k},
    )
