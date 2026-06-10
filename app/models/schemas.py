"""Pydantic schemas for request and response payloads."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query from the Chrome extension."""

    video_id: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="YouTube video ID (e.g. 'hVIfPT0RWlo')",
        examples=["hVIfPT0RWlo"],
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question about the video",
    )
    language: str = Field(
        default="en",
        max_length=10,
        description="BCP-47 language code for the transcript (e.g. 'en', 'hi')",
    )


class QueryResponse(BaseModel):
    """Non-streaming response returned to the client."""

    video_id: str
    query: str
    answer: str


class HealthResponse(BaseModel):
    """Simple liveness/readiness probe response."""

    status: str = "ok"


class TranscriptLanguage(BaseModel):
    """A single available transcript language for a video."""

    code: str = Field(description="BCP-47 language code, e.g. 'en', 'hi'")
    name: str = Field(description="Human-readable language name, e.g. 'English'")
    is_generated: bool = Field(description="True if auto-generated, False if manually captioned")


class LanguagesResponse(BaseModel):
    """Available transcript languages for a YouTube video."""

    video_id: str
    languages: list[TranscriptLanguage]
