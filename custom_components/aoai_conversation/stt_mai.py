"""MAI-Transcribe-2-Streaming speech-to-text transport."""

from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import json
from typing import Any
from urllib.parse import urlparse

import aiohttp

from homeassistant.exceptions import HomeAssistantError

from .stt_backend import STTRequest

_EVENT_TRANSCRIPTION = "conversation.item.input_audio_transcription."
_SESSION_TIMEOUT = 30.0


def mai_realtime_url(base: str) -> str:
    """Return the MAI streaming WebSocket URL for a Foundry resource root."""
    endpoint = urlparse(base.strip())
    if (
        endpoint.scheme not in {"https", "wss"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.path not in {"", "/"}
        or endpoint.params
        or endpoint.query
        or endpoint.fragment
    ):
        raise HomeAssistantError(
            "MAI streaming endpoint must be an HTTPS or WSS Foundry resource root URL"
        )
    return f"wss://{endpoint.netloc}/mai/v1/realtime?intent=transcription"


async def _async_receive_event(
    websocket: aiohttp.ClientWebSocketResponse,
) -> dict[str, Any]:
    """Receive one MAI streaming event and surface protocol errors."""
    message = await websocket.receive()
    if message.type is not aiohttp.WSMsgType.TEXT:
        detail = (
            str(websocket.exception())
            if message.type is aiohttp.WSMsgType.ERROR
            else message.type.name
        )
        raise HomeAssistantError(f"MAI streaming connection closed: {detail}")

    try:
        event = json.loads(message.data)
    except json.JSONDecodeError as err:
        raise HomeAssistantError("MAI streaming returned invalid JSON") from err
    if not isinstance(event, dict):
        raise HomeAssistantError("MAI streaming returned an invalid event")
    if event.get("type") in {"error", f"{_EVENT_TRANSCRIPTION}failed"}:
        raise HomeAssistantError(
            f"MAI streaming request failed: {event.get('error', event)}"
        )
    return event


async def _async_wait_for_event(
    websocket: aiohttp.ClientWebSocketResponse, event_type: str
) -> None:
    """Wait for a MAI streaming session event."""
    try:
        async with asyncio.timeout(_SESSION_TIMEOUT):
            while (await _async_receive_event(websocket)).get("type") != event_type:
                pass
    except TimeoutError as err:
        raise HomeAssistantError(
            f"MAI streaming timed out waiting for {event_type}"
        ) from err


async def async_transcribe(
    client: aiohttp.ClientSession, request: STTRequest
) -> str | None:
    """Forward PCM audio to MAI and return its completed transcript."""
    if not request.deployment:
        raise HomeAssistantError("MAI streaming requires a deployment name")

    websocket_url = mai_realtime_url(request.endpoint)
    language = request.language.partition("-")[0].lower()

    try:
        async with client.ws_connect(
            websocket_url, headers={"api-key": request.api_key}
        ) as websocket:
            await _async_wait_for_event(websocket, "session.created")
            await websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": request.sample_rate,
                                },
                                "transcription": {
                                    "model": request.deployment,
                                    "language": language or None,
                                },
                                "turn_detection": None,
                                "noise_reduction": None,
                            }
                        },
                    },
                }
            )
            await _async_wait_for_event(websocket, "session.updated")

            loop = asyncio.get_running_loop()
            completion: asyncio.Future[str] = loop.create_future()

            async def receive_transcript() -> None:
                """Receive transcript events until the committed segment completes."""
                try:
                    while True:
                        event = await _async_receive_event(websocket)
                        if event.get("type") != f"{_EVENT_TRANSCRIPTION}completed":
                            continue
                        transcript = event.get("transcript")
                        if not isinstance(transcript, str):
                            raise HomeAssistantError(
                                "MAI streaming completed without a transcript"
                            )
                        completion.set_result(transcript)
                        return
                except Exception as err:
                    if not completion.done():
                        completion.set_exception(err)

            receiver = asyncio.create_task(
                receive_transcript(), name="mai-streaming-stt-receiver"
            )
            audio_sent = False
            try:
                async for chunk in request.audio_stream:
                    if not chunk:
                        continue
                    audio_sent = True
                    await websocket.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )

                if not audio_sent:
                    return None

                await websocket.send_json({"type": "input_audio_buffer.commit"})
                try:
                    return await asyncio.wait_for(completion, timeout=_SESSION_TIMEOUT)
                except TimeoutError as err:
                    raise HomeAssistantError(
                        "MAI streaming timed out waiting for the final transcript"
                    ) from err
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
    except aiohttp.ClientError as err:
        raise HomeAssistantError(f"MAI streaming request failed: {err}") from err
