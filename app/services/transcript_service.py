"""
Transcript service — fetches, splits, embeds, and caches a YouTube transcript
into a per-video-id in-memory vector store so that repeated queries on the same
video do NOT re-embed the transcript.

Embeddings are produced via the HuggingFace Inference API (no local model
weights) so the deployment stays well within Vercel's 500 MB Lambda limit.
"""

import logging
from threading import Lock
from typing import Dict

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from app.config import Settings

logger = logging.getLogger(__name__)


class _HFInferenceEmbeddings(Embeddings):
    """Calls the HuggingFace feature-extraction API; no local model weights.

    Uses huggingface_hub.InferenceClient directly — avoids the deprecated
    langchain-community wrapper and keeps the Vercel bundle under 500 MB.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = InferenceClient(token=api_key)
        self._model = model

    def _embed(self, text: str) -> list[float]:
        raw = self._client.feature_extraction(text, model=self._model)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        # Sentence models return a 1-D list; token-level models return 2-D.
        # Apply mean pooling over the token dimension when needed.
        if raw and isinstance(raw[0], list):
            n = len(raw)
            return [sum(row[i] for row in raw) / n for i in range(len(raw[0]))]
        return raw

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# In-process cache: video_id → InMemoryVectorStore
_store_cache: Dict[str, InMemoryVectorStore] = {}
_cache_lock = Lock()


def _build_embeddings(settings: Settings) -> _HFInferenceEmbeddings:
    """Use HuggingFace Inference API for embeddings — no local model weights."""
    return _HFInferenceEmbeddings(
        api_key=settings.huggingface_api_key,
        model=settings.embedding_model,
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
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

    with _cache_lock:
        _store_cache[cache_key] = vector_store

    logger.info(
        "Vector store for '%s' built with %d chunks.", video_id, len(chunks)
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k},
    )
