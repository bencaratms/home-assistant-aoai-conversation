"""Tests for the Azure AI Speech REST helpers (speech.py)."""

import asyncio
from collections.abc import AsyncIterator
import json
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import httpx
import pytest

from custom_components.aoai_conversation.client import (
    async_create_conversation,
    async_delete_conversation,
)
from custom_components.aoai_conversation.speech import (
    async_list_voices,
    async_synthesize,
    build_ssml,
    speech_url,
)
from custom_components.aoai_conversation.stt_backend import STTRequest
from custom_components.aoai_conversation.stt_buffered import (
    async_transcribe as async_buffered_transcribe,
)
from custom_components.aoai_conversation.stt_mai import (
    async_transcribe as async_mai_transcribe,
    mai_realtime_url,
)
from custom_components.aoai_conversation.stt_realtime import (
    async_transcribe as async_realtime_transcribe,
    speech_realtime_url,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

BASE = "https://lohmannio.cognitiveservices.azure.com/"


@pytest.mark.parametrize(
    ("base", "kind", "expected"),
    [
        # Custom-subdomain host (the common Azure AI Foundry case) -> service prefix.
        (
            BASE,
            "tts",
            "https://lohmannio.cognitiveservices.azure.com/tts/cognitiveservices/v1",
        ),
        (
            BASE,
            "voices",
            "https://lohmannio.cognitiveservices.azure.com/tts/cognitiveservices/voices/list",
        ),
        (
            BASE,
            "stt",
            "https://lohmannio.cognitiveservices.azure.com/stt/speech/recognition/conversation/cognitiveservices/v1",
        ),
        # Regional hosts encode the service -> no prefix.
        (
            "https://westeurope.tts.speech.microsoft.com/",
            "tts",
            "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1",
        ),
        (
            "https://westeurope.stt.speech.microsoft.com/",
            "stt",
            "https://westeurope.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
        ),
    ],
)
def test_speech_url(base: str, kind: str, expected: str) -> None:
    """URLs are built with the correct host-aware suffixes."""
    assert speech_url(base, kind) == expected
    assert speech_url(base.rstrip("/"), kind) == expected


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        (
            "https://resource.services.ai.azure.com/",
            "wss://resource.services.ai.azure.com/mai/v1/realtime?intent=transcription",
        ),
        (
            "wss://resource.services.ai.azure.com",
            "wss://resource.services.ai.azure.com/mai/v1/realtime?intent=transcription",
        ),
    ],
)
def test_mai_realtime_url(base: str, expected: str) -> None:
    """MAI uses the Foundry resource root and a fixed realtime path."""
    assert mai_realtime_url(base) == expected


@pytest.mark.parametrize(
    "base",
    [
        "https://resource.services.ai.azure.com/api/projects/project",
        "https://resource.services.ai.azure.com/?query=value",
        "http://resource.services.ai.azure.com",
    ],
)
def test_mai_realtime_url_rejects_non_resource_roots(base: str) -> None:
    """MAI requires a clean HTTPS or WSS Foundry resource root."""
    with pytest.raises(HomeAssistantError, match="resource root"):
        mai_realtime_url(base)


def test_build_ssml_basic() -> None:
    """A minimal SSML document contains the voice and escaped text."""
    ssml = build_ssml("Hallo & <Welt>", "de-DE-KatjaNeural", "de-DE")
    assert "<voice name='de-DE-KatjaNeural'>" in ssml
    assert "xml:lang='de-DE'" in ssml
    assert "Hallo &amp; &lt;Welt&gt;" in ssml
    assert "<prosody" not in ssml
    assert "express-as" not in ssml


def test_build_ssml_prosody_and_style() -> None:
    """Rate/pitch/style wrap the text in the right SSML elements."""
    ssml = build_ssml(
        "Hi",
        "de-DE-KatjaNeural",
        "de-DE",
        rate="+10%",
        pitch="-2st",
        style="cheerful",
    )
    assert 'rate="+10%"' in ssml
    assert 'pitch="-2st"' in ssml
    assert 'style="cheerful"' in ssml
    assert "mstts:express-as" in ssml


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _MockWebSocket:
    """Small WebSocket mock that queues a completion after commit."""

    def __init__(self) -> None:
        self.events = [
            {"type": "session.created"},
            {"type": "session.updated"},
        ]
        self.sent: list[dict] = []
        self._event_available = asyncio.Event()

    async def __aenter__(self) -> _MockWebSocket:
        """Enter the mock connection context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit the mock connection context."""

    async def send_json(self, data: dict) -> None:
        """Record a client event and complete after its commit."""
        self.sent.append(data)
        if data["type"] == "input_audio_buffer.commit":
            self.events.append(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Turn on the kitchen light",
                }
            )
            self._event_available.set()

    async def receive(self) -> SimpleNamespace:
        """Return queued server events."""
        while not self.events:
            await self._event_available.wait()
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT, data=json.dumps(self.events.pop(0))
        )

    def exception(self) -> None:
        """Match the aiohttp WebSocket error API."""
        return None


class _MockWebSocketClient:
    """Client mock exposing aiohttp's ws_connect context-manager contract."""

    def __init__(self) -> None:
        self.websocket = _MockWebSocket()
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def ws_connect(self, url: str, *, headers: dict[str, str]) -> _MockWebSocket:
        """Return the reusable mock WebSocket."""
        self.url = url
        self.headers = headers
        return self.websocket


class _MockSpeechWebSocket:
    """WebSocket mock that records Azure Speech text and binary frames."""

    def __init__(self) -> None:
        self.events = [
            (
                "speech.phrase",
                {
                    "RecognitionStatus": "Success",
                    "DisplayText": "Licht an",
                },
            ),
            ("turn.end", {}),
        ]
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def __aenter__(self) -> _MockSpeechWebSocket:
        """Enter the mock connection context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit the mock connection context."""

    async def send_str(self, data: str) -> None:
        """Record a control frame."""
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        """Record a binary frame."""
        self.sent_bytes.append(data)

    async def receive(self) -> SimpleNamespace:
        """Return a queued Speech protocol event."""
        path, body = self.events.pop(0)
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=f"Path: {path}\r\n\r\n{json.dumps(body)}",
        )

    def exception(self) -> None:
        """Match the aiohttp WebSocket error API."""
        return None


class _MockSpeechWebSocketClient:
    """Client mock exposing an Azure Speech WebSocket connection."""

    def __init__(self) -> None:
        self.websocket = _MockSpeechWebSocket()
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def ws_connect(self, url: str, *, headers: dict[str, str]) -> _MockSpeechWebSocket:
        """Return the reusable mock WebSocket."""
        self.url = url
        self.headers = headers
        return self.websocket


async def _audio_chunks() -> AsyncIterator[bytes]:
    """Yield two PCM chunks like Home Assistant's STT stream."""
    yield b"\x01\x00"
    yield b"\x02\x00"


async def test_async_mai_transcribe() -> None:
    """MAI forwards chunks immediately and returns the committed transcript."""
    client = _MockWebSocketClient()

    transcript = await async_mai_transcribe(
        client,  # type: ignore[arg-type]
        STTRequest(
            endpoint="https://resource.services.ai.azure.com/",
            api_key="mai-key",
            deployment="mai-transcribe-deployment",
            language="de-DE",
            audio_stream=_audio_chunks(),
        ),
    )

    assert transcript == "Turn on the kitchen light"
    assert client.url == (
        "wss://resource.services.ai.azure.com/mai/v1/realtime?intent=transcription"
    )
    assert client.headers == {"api-key": "mai-key"}
    assert client.websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {
                            "model": "mai-transcribe-deployment",
                            "language": "de",
                        },
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        },
        {
            "type": "input_audio_buffer.append",
            "audio": "AQA=",
        },
        {
            "type": "input_audio_buffer.append",
            "audio": "AgA=",
        },
        {"type": "input_audio_buffer.commit"},
    ]


async def test_async_synthesize_success(hass: HomeAssistant) -> None:
    """Synthesis returns the raw audio bytes and sends SSML headers."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = request.content
        return httpx.Response(200, content=b"AUDIO")

    audio = await async_synthesize(
        _mock_client(handler),
        BASE,
        "key",
        "<speak/>",
        "audio-24khz-48kbitrate-mono-mp3",
    )

    assert audio == b"AUDIO"
    assert captured["url"].endswith("/cognitiveservices/v1")
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "key"
    assert captured["headers"]["Content-Type"] == "application/ssml+xml"
    assert (
        captured["headers"]["X-Microsoft-OutputFormat"]
        == "audio-24khz-48kbitrate-mono-mp3"
    )


async def test_async_synthesize_error(hass: HomeAssistant) -> None:
    """An HTTP error is surfaced as a HomeAssistantError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="denied")

    with pytest.raises(HomeAssistantError):
        await async_synthesize(_mock_client(handler), BASE, "key", "<speak/>", "fmt")


async def test_async_list_voices(hass: HomeAssistant) -> None:
    """Voice listing returns the parsed JSON list."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"ShortName": "de-DE-KatjaNeural", "LocalName": "Katja"},
                {"ShortName": "en-US-JennyNeural", "LocalName": "Jenny"},
            ],
        )

    voices = await async_list_voices(_mock_client(handler), BASE, "key")

    assert [v["ShortName"] for v in voices] == [
        "de-DE-KatjaNeural",
        "en-US-JennyNeural",
    ]


async def test_async_buffered_transcribe_success(hass: HomeAssistant) -> None:
    """Recognition returns DisplayText for a successful status."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200, json={"RecognitionStatus": "Success", "DisplayText": "Licht an"}
        )

    text = await async_buffered_transcribe(
        _mock_client(handler),
        STTRequest(
            endpoint=BASE,
            api_key="key",
            language="de-DE",
            audio_stream=_audio_chunks(),
        ),
    )

    assert text == "Licht an"
    assert "language=de-DE" in captured["url"]
    assert "/speech/recognition/conversation/cognitiveservices/v1" in captured["url"]


async def test_async_buffered_transcribe_no_match(hass: HomeAssistant) -> None:
    """A non-success status yields None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"RecognitionStatus": "NoMatch"})

    text = await async_buffered_transcribe(
        _mock_client(handler),
        STTRequest(
            endpoint=BASE,
            api_key="key",
            language="de-DE",
            audio_stream=_audio_chunks(),
        ),
    )

    assert text is None


def test_speech_realtime_url() -> None:
    """Realtime Speech converts the configured HTTPS endpoint to WSS."""
    assert speech_realtime_url(BASE, "de-DE") == (
        "wss://lohmannio.cognitiveservices.azure.com/stt/speech/recognition/"
        "conversation/cognitiveservices/v1?language=de-DE&format=detailed"
    )


async def test_async_realtime_transcribe() -> None:
    """Realtime Speech forwards framed PCM and returns its final phrase."""
    client = _MockSpeechWebSocketClient()
    transcript = await async_realtime_transcribe(
        client,  # type: ignore[arg-type]
        STTRequest(
            endpoint=BASE,
            api_key="key",
            language="de-DE",
            audio_stream=_audio_chunks(),
        ),
    )

    assert transcript == "Licht an"
    assert client.headers == {"Ocp-Apim-Subscription-Key": "key"}
    assert len(client.websocket.sent_text) == 2
    assert len(client.websocket.sent_bytes) == 4


async def test_async_create_conversation(hass: HomeAssistant) -> None:
    """Creating a conversation posts to the project endpoint and returns the id."""

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": "conv_abc123"})

    with patch(
        "custom_components.aoai_conversation.client.get_async_client",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        conv_id = await async_create_conversation(
            hass,
            "https://res.services.ai.azure.com/api/projects/proj",
            "sk-key",
        )

    assert conv_id == "conv_abc123"
    assert captured["url"].endswith("/api/projects/proj/openai/v1/conversations")
    assert captured["auth"] == "Bearer sk-key"


async def test_async_delete_conversation_swallows_errors(hass: HomeAssistant) -> None:
    """Deleting a conversation is best-effort and never raises."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="gone")

    with patch(
        "custom_components.aoai_conversation.client.get_async_client",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        # Should not raise despite the 404.
        await async_delete_conversation(
            hass,
            "https://res.services.ai.azure.com/api/projects/proj",
            "sk-key",
            "conv_abc123",
        )
