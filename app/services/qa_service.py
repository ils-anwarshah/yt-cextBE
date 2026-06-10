"""
QA service — formats the prompt, calls the HuggingFace inference API,
and returns both a complete answer and an async token stream.
"""

import logging
from typing import AsyncIterator

from huggingface_hub import AsyncInferenceClient
from langchain_core.prompts import PromptTemplate

from app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful assistant that answers questions strictly based \
on the provided context from a YouTube video transcript.

Rules:
- Only use information present in the context below.
- If the answer is not in the context or different from greeting, respond with: "I don't know based on the provided transcript."
- Be concise and accurate.

Context:
{context}

Question: {query}

Answer:""",
    input_variables=["context", "query"],
)


def _build_context(docs: list) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_client(settings: Settings) -> AsyncInferenceClient:
    """Create an async HuggingFace inference client."""
    return AsyncInferenceClient(token=settings.huggingface_api_key)


# ---------------------------------------------------------------------------
# Non-streaming answer
# ---------------------------------------------------------------------------
async def get_answer(
    query: str,
    docs: list,
    settings: Settings,
) -> str:
    """
    Return the full model answer for the given query and retrieved docs.

    Args:
        query:    The user question.
        docs:     Documents returned by the retriever.
        settings: Application settings.

    Returns:
        The model's text response.
    """
    context = _build_context(docs)
    prompt_text = _PROMPT_TEMPLATE.format(context=context, query=query)

    client = _build_client(settings)
    try:
        response = await client.chat.completions.create(
            model=settings.inference_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=settings.max_new_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("HuggingFace inference error: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Streaming answer (Server-Sent Events)
# ---------------------------------------------------------------------------
async def stream_answer(
    query: str,
    docs: list,
    settings: Settings,
) -> AsyncIterator[str]:
    """
    Yield individual token strings as they arrive from the model.

    Args:
        query:    The user question.
        docs:     Documents returned by the retriever.
        settings: Application settings.

    Yields:
        Token strings from the model stream.
    """
    context = _build_context(docs)
    prompt_text = _PROMPT_TEMPLATE.format(context=context, query=query)

    client = _build_client(settings)
    try:
        stream = await client.chat.completions.create(
            model=settings.inference_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=settings.max_new_tokens,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as exc:
        logger.error("HuggingFace streaming error: %s", exc)
        raise
