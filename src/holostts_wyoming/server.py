"""Wyoming event handler: describe, synthesize, and streamed synthesize-start/chunk/stop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from . import __version__
from .engines.base import Engine
from .pipeline import split_sentences

_LOGGER = logging.getLogger(__name__)
_WIDTH, _CHANNELS = 2, 1
_HOLOS = Attribution(name="patriotyk", url="https://huggingface.co/patriotyk/HolosTTS")


@dataclass
class Settings:
    voice: str
    speed: float


def make_info(voices: list[str]) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="holostts",
                description="Ukrainian text to speech with HolosTTS",
                attribution=_HOLOS,
                installed=True,
                version=__version__,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(name=v, description=v, attribution=_HOLOS, installed=True, version=None, languages=["uk"])
                    for v in voices
                ],
            )
        ]
    )


class HolosHandler(AsyncEventHandler):
    def __init__(
        self,
        info: Info,
        engine: Engine,
        phonemizer: Callable[[str], list[str]],
        settings: Settings,
        lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info = info
        self._engine = engine
        self._phonemizer = phonemizer
        self._settings = settings
        self._lock = lock
        self._stream_buffer: str | None = None
        self._stream_voice = settings.voice

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self._info.event())
            elif SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self._stream_voice = self._voice(start.voice.name if start.voice else None)
                self._stream_buffer = ""
                await self._audio_start()
            elif SynthesizeChunk.is_type(event.type) and self._stream_buffer is not None:
                self._stream_buffer += SynthesizeChunk.from_event(event).text
                sentences = split_sentences(self._stream_buffer)
                if len(sentences) > 1:
                    # the last piece may be an unfinished sentence (or "21" of "21.5"): keep it
                    head, self._stream_buffer = sentences[:-1], sentences[-1]
                    await self._speak(" ".join(head), self._stream_voice)
            elif SynthesizeStop.is_type(event.type) and self._stream_buffer is not None:
                rest, self._stream_buffer = self._stream_buffer, None
                await self._speak(rest, self._stream_voice)
                await self.write_event(AudioStop().event())
                await self.write_event(SynthesizeStopped().event())
            elif Synthesize.is_type(event.type):
                if self._stream_buffer is not None:
                    return True  # sent alongside a stream for older servers
                synthesize = Synthesize.from_event(event)
                voice = self._voice(synthesize.voice.name if synthesize.voice else None)
                await self._audio_start()
                await self._speak(synthesize.text, voice)
                await self.write_event(AudioStop().event())
        except Exception as error:  # keep the connection; report to the client
            _LOGGER.exception("Error handling %s", event.type)
            await self.write_event(Error(text=str(error), code=error.__class__.__name__).event())
        return True

    def _voice(self, name: str | None) -> str:
        if name and name in self._engine.voices():
            return name
        if name:
            _LOGGER.warning("Unknown voice %r, using %s", name, self._settings.voice)
        return self._settings.voice

    async def _audio_start(self) -> None:
        await self.write_event(AudioStart(rate=self._engine.sample_rate, width=_WIDTH, channels=_CHANNELS).event())

    async def _speak(self, text: str, voice: str) -> None:
        _LOGGER.debug("Text: %s", text)
        for phonemes in self._phonemizer(text):
            _LOGGER.debug("Phonemes: %s", phonemes)
            async for chunk in self._synthesize(phonemes, voice):
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                await self.write_event(
                    AudioChunk(rate=self._engine.sample_rate, width=_WIDTH, channels=_CHANNELS, audio=pcm).event()
                )

    async def _synthesize(self, phonemes: str, voice: str) -> AsyncIterator[np.ndarray]:
        """Run the engine in a thread, one synthesis at a time, handing chunks over as they come."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def produce() -> None:
            try:
                for chunk in self._engine.stream(phonemes, voice, self._settings.speed):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        async with self._lock:
            worker = loop.run_in_executor(None, produce)
            while (item := await queue.get()) is not done:
                if isinstance(item, Exception):
                    raise item
                yield item
            await worker
