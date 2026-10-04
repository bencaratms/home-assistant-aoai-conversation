"""Shared request contract for speech-to-text transports."""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class STTRequest:
    """One Home Assistant speech-to-text request."""

    endpoint: str
    api_key: str
    language: str
    audio_stream: AsyncIterable[bytes]
    deployment: str | None = None
    channels: int = 1
    sample_width: int = 2
    sample_rate: int = 16000
