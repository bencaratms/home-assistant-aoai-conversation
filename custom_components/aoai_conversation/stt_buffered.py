"""Buffered Azure Speech REST speech-to-text transport."""

from __future__ import annotations

import io
import wave

import httpx

from homeassistant.exceptions import HomeAssistantError

from .const import LOGGER
from .speech import speech_url
from .stt_backend import STTRequest

_STT_SUCCESS = "Success"


async def async_transcribe(
    client: httpx.AsyncClient, request: STTRequest
) -> str | None:
    """Buffer PCM audio, submit it as WAV, and return the final transcript."""
    audio_bytes = bytearray()
    async for chunk in request.audio_stream:
        audio_bytes.extend(chunk)

    if not audio_bytes:
        return None

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(request.channels)
        wav_file.setsampwidth(request.sample_width)
        wav_file.setframerate(request.sample_rate)
        wav_file.writeframes(bytes(audio_bytes))

    try:
        response = await client.post(
            speech_url(request.endpoint, "stt"),
            params={"language": request.language, "format": "detailed"},
            headers={
                "Ocp-Apim-Subscription-Key": request.api_key,
                "Content-Type": (
                    f"audio/wav; codecs=audio/pcm; samplerate={request.sample_rate}"
                ),
                "Accept": "application/json",
                "User-Agent": "home-assistant-aoai-conversation",
            },
            content=wav_buffer.getvalue(),
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        raise HomeAssistantError(
            f"Azure Speech STT request failed ({err.response.status_code}): "
            f"{err.response.text}"
        ) from err
    except httpx.HTTPError as err:
        raise HomeAssistantError(f"Azure Speech STT request failed: {err}") from err

    result = response.json()
    status = result.get("RecognitionStatus")
    if status != _STT_SUCCESS:
        LOGGER.debug("Azure Speech STT non-success status: %s", status)
        return None
    return result.get("DisplayText")
