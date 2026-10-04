"""Azure Speech realtime WebSocket speech-to-text transport."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import struct
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import uuid4

import aiohttp

from homeassistant.exceptions import HomeAssistantError

from .speech import speech_url
from .stt_backend import STTRequest

_SESSION_TIMEOUT = 30.0
_MAX_HEADER_LENGTH = 0x7FFF
_HTTP_OK = 200


def speech_realtime_url(base: str, language: str) -> str:
    """Return the Azure Speech conversation-recognition WebSocket URL."""
    endpoint = urlsplit(speech_url(base, "stt"))
    return urlunsplit(
        (
            "wss",
            endpoint.netloc,
            endpoint.path,
            urlencode({"language": language, "format": "detailed"}),
            "",
        )
    )


def _headers(path: str, request_id: str, content_type: str | None = None) -> bytes:
    """Build Azure Speech protocol headers."""
    fields = [
        f"Path: {path}",
        f"X-RequestId: {request_id}",
    ]
    if content_type:
        fields.append(f"Content-Type: {content_type}")
    return "\r\n".join(fields).encode("ascii")


def _binary_message(headers: bytes, body: bytes = b"") -> bytes:
    """Frame a binary Azure Speech WebSocket message."""
    if len(headers) > _MAX_HEADER_LENGTH:
        raise ValueError("Azure Speech WebSocket headers are too large")
    return struct.pack(">H", len(headers)) + headers + body


def _wav_header(request: STTRequest) -> bytes:
    """Create a streaming PCM WAV header with an unknown final data length."""
    byte_rate = request.sample_rate * request.channels * request.sample_width
    block_align = request.channels * request.sample_width
    bits_per_sample = request.sample_width * 8
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVEfmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                request.channels,
                request.sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ),
            b"data",
            struct.pack("<I", 0xFFFFFFFF),
        )
    )


def _parse_message(message: str) -> tuple[str, dict[str, Any]]:
    """Parse a text Azure Speech event into its protocol path and JSON body."""
    headers, _, body = message.partition("\r\n\r\n")
    path = ""
    for header in headers.split("\r\n"):
        name, separator, value = header.partition(":")
        if separator and name.lower() == "path":
            path = value.strip().lower()
            break
    if not path:
        raise HomeAssistantError("Azure Speech realtime response has no Path header")
    if not body:
        return path, {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as err:
        raise HomeAssistantError("Azure Speech realtime returned invalid JSON") from err
    if not isinstance(payload, dict):
        raise HomeAssistantError("Azure Speech realtime returned an invalid event")
    return path, payload


async def async_transcribe(
    client: aiohttp.ClientSession, request: STTRequest
) -> str | None:
    """Forward PCM audio over Azure Speech WebSocket and return its final transcript."""
    request_id = uuid4().hex
    url = speech_realtime_url(request.endpoint, request.language)
    speech_config = {
        "context": {
            "system": {
                "name": "home-assistant-aoai-conversation",
                "version": "1.0",
                "build": "Python",
                "lang": "Python",
            },
            "audio": {
                "source": {
                    "samplerate": request.sample_rate,
                    "bitspersample": request.sample_width * 8,
                    "channelcount": request.channels,
                }
            },
        }
    }
    speech_context = {"phraseDetection": {"mode": "CONVERSATION"}}

    try:
        async with client.ws_connect(
            url,
            headers={"Ocp-Apim-Subscription-Key": request.api_key},
        ) as websocket:
            await websocket.send_str(
                _headers("speech.config", request_id, "application/json").decode()
                + "\r\n\r\n"
                + json.dumps(speech_config, separators=(",", ":"))
            )
            await websocket.send_str(
                _headers("speech.context", request_id, "application/json").decode()
                + "\r\n\r\n"
                + json.dumps(speech_context, separators=(",", ":"))
            )

            wav_headers = _headers("audio", request_id, "audio/x-wav")
            audio_headers = _headers("audio", request_id)
            await websocket.send_bytes(
                _binary_message(wav_headers, _wav_header(request))
            )

            loop = asyncio.get_running_loop()
            completion: asyncio.Future[str | None] = loop.create_future()

            async def receive_transcript() -> None:
                """Receive partial/final Azure Speech events until the turn ends."""
                transcript: str | None = None
                try:
                    while True:
                        message = await websocket.receive()
                        if message.type is not aiohttp.WSMsgType.TEXT:
                            detail = (
                                str(websocket.exception())
                                if message.type is aiohttp.WSMsgType.ERROR
                                else message.type.name
                            )
                            raise HomeAssistantError(
                                f"Azure Speech realtime connection closed: {detail}"
                            )
                        path, payload = _parse_message(message.data)
                        if path == "speech.phrase":
                            if payload.get("RecognitionStatus") == "Success":
                                display_text = payload.get("DisplayText")
                                if isinstance(display_text, str):
                                    transcript = display_text
                        elif path == "turn.end":
                            completion.set_result(transcript)
                            return
                        elif path == "response" and payload.get("status") != _HTTP_OK:
                            raise HomeAssistantError(
                                f"Azure Speech realtime request failed: {payload}"
                            )
                except Exception as err:
                    if not completion.done():
                        completion.set_exception(err)

            receiver = asyncio.create_task(
                receive_transcript(), name="azure-speech-realtime-stt-receiver"
            )
            audio_sent = False
            try:
                async for chunk in request.audio_stream:
                    if not chunk:
                        continue
                    audio_sent = True
                    await websocket.send_bytes(_binary_message(audio_headers, chunk))

                if not audio_sent:
                    return None

                await websocket.send_bytes(_binary_message(audio_headers))
                try:
                    return await asyncio.wait_for(completion, timeout=_SESSION_TIMEOUT)
                except TimeoutError as err:
                    raise HomeAssistantError(
                        "Azure Speech realtime timed out waiting for the final "
                        "transcript"
                    ) from err
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
    except aiohttp.ClientError as err:
        raise HomeAssistantError(
            f"Azure Speech realtime request failed: {err}"
        ) from err
