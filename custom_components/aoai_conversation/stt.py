"""Speech to text support via Azure AI Speech."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging
from typing import TYPE_CHECKING, override

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_STT_API_KEY,
    CONF_STT_BACKEND,
    CONF_STT_DEPLOYMENT,
    CONF_STT_ENDPOINT,
    CONF_STT_LANGUAGE,
    DEFAULT_STT_LANGUAGE,
    DOMAIN,
    STT_BACKEND_AZURE_SPEECH,
    STT_BACKEND_AZURE_SPEECH_REALTIME,
    STT_BACKEND_MAI_STREAMING,
)
from .stt_backend import STTRequest
from .stt_buffered import async_transcribe as async_buffered_transcribe
from .stt_mai import async_transcribe as async_mai_transcribe
from .stt_realtime import async_transcribe as async_realtime_transcribe

if TYPE_CHECKING:
    from . import OpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [AzureSpeechSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class AzureSpeechSTTEntity(stt.SpeechToTextEntity, Entity):
    """Azure AI Speech speech-to-text entity."""

    def __init__(self, entry: OpenAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Azure AI Speech",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        # Azure Speech supports 100+ locales; expose a broad common set. The
        # configured language is used at recognition time regardless.
        return [
            "ar-SA",
            "bg-BG",
            "ca-ES",
            "cs-CZ",
            "da-DK",
            "de-DE",
            "de-AT",
            "de-CH",
            "el-GR",
            "en-US",
            "en-GB",
            "en-AU",
            "en-CA",
            "en-IN",
            "es-ES",
            "es-MX",
            "et-EE",
            "fi-FI",
            "fr-FR",
            "fr-CA",
            "he-IL",
            "hi-IN",
            "hr-HR",
            "hu-HU",
            "id-ID",
            "it-IT",
            "ja-JP",
            "ko-KR",
            "lt-LT",
            "lv-LV",
            "ms-MY",
            "nb-NO",
            "nl-NL",
            "pl-PL",
            "pt-BR",
            "pt-PT",
            "ro-RO",
            "ru-RU",
            "sk-SK",
            "sl-SI",
            "sv-SE",
            "th-TH",
            "tr-TR",
            "uk-UA",
            "vi-VN",
            "zh-CN",
            "zh-HK",
            "zh-TW",
        ]

    @property
    @override
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV]

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    @override
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream to the Azure Speech STT service."""
        endpoint = self.subentry.data.get(CONF_STT_ENDPOINT)
        api_key = self.subentry.data.get(CONF_STT_API_KEY)
        if not endpoint or not api_key:
            _LOGGER.error(
                "Azure Speech STT is not configured: set the endpoint URI and "
                "API key for the '%s' entity",
                self.subentry.title,
            )
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        backend = self.subentry.data.get(CONF_STT_BACKEND, STT_BACKEND_AZURE_SPEECH)
        language = self.subentry.data.get(CONF_STT_LANGUAGE) or (
            metadata.language if backend != STT_BACKEND_MAI_STREAMING else ""
        )
        if not language and backend != STT_BACKEND_MAI_STREAMING:
            language = DEFAULT_STT_LANGUAGE

        request = STTRequest(
            endpoint=endpoint,
            api_key=api_key,
            language=language,
            deployment=self.subentry.data.get(CONF_STT_DEPLOYMENT),
            audio_stream=stream,
            channels=metadata.channel.value,
            sample_width=metadata.bit_rate.value // 8,
            sample_rate=metadata.sample_rate.value,
        )

        try:
            if backend == STT_BACKEND_AZURE_SPEECH:
                text = await async_buffered_transcribe(
                    get_async_client(self.hass), request
                )
            elif backend == STT_BACKEND_AZURE_SPEECH_REALTIME:
                text = await async_realtime_transcribe(
                    async_get_clientsession(self.hass), request
                )
            elif backend == STT_BACKEND_MAI_STREAMING:
                text = await async_mai_transcribe(
                    async_get_clientsession(self.hass), request
                )
            else:
                _LOGGER.error(
                    "Unknown STT backend '%s' for the '%s' entity",
                    backend,
                    self.subentry.title,
                )
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        except HomeAssistantError as err:
            _LOGGER.error("Error during %s STT: %s", backend, err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if text:
            return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
