import asyncio
import threading
import time
from functools import partial

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from holostts_wyoming.pipeline import split_sentences
from holostts_wyoming.server import HolosHandler, Settings, make_info


class FakeEngine:
    sample_rate = 24000

    def __init__(self, delay=0.0, chunks=1):
        self.calls = []
        self.delay = delay
        self.chunks = chunks
        self.yielded = 0
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def voices(self):
        return ["Speaker_0", "Speaker_1"]

    def stream(self, phonemes, voice, speed):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            for i in range(self.chunks):
                time.sleep(self.delay)
                if i == 0:
                    self.calls.append((phonemes, voice))
                self.yielded += 1
                yield np.zeros(2400, dtype=np.float32)
        finally:
            with self._guard:
                self.active -= 1


def fake_phonemizer(text):
    # same contract as pipeline.make_phonemizer: one string per sentence, silent ones dropped
    return [s for s in split_sentences(text) if any(c.isalpha() for c in s)]


async def serve(engine):
    info = make_info(engine.voices())
    factory = partial(HolosHandler, info, engine, fake_phonemizer, Settings(voice="Speaker_0", speed=1.0), asyncio.Lock())

    async def on_client(reader, writer):
        await factory(reader, writer).run()

    server = await asyncio.start_server(on_client, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def read_until(client, stop_types):
    events = []
    while True:
        event = await asyncio.wait_for(client.read_event(), 5)
        events.append(event)
        if event.type in stop_types:
            return events


def run(coro):
    return asyncio.run(coro)


def test_describe_reports_streaming_and_voices():
    async def go():
        server, port = await serve(FakeEngine())
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            info = Info.from_event(await client.read_event())
        server.close()
        return info

    info = run(go())
    assert info.tts[0].name == "holostts"
    assert info.tts[0].supports_synthesize_streaming is True
    assert [v.name for v in info.tts[0].voices] == ["Speaker_0", "Speaker_1"]
    assert info.tts[0].voices[0].languages == ["uk"]


def test_synthesize_one_audio_stream_per_request():
    engine = FakeEngine()

    async def go():
        server, port = await serve(engine)
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="Привіт. Як справи?").event())
            events = await read_until(client, {AudioStop().event().type})
        server.close()
        return events

    events = run(go())
    assert [e.type for e in events] == ["audio-start", "audio-chunk", "audio-chunk", "audio-stop"]
    start = AudioStart.from_event(events[0])
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)
    assert len(engine.calls) == 2


def test_streaming_speaks_complete_sentences_early():
    engine = FakeEngine()

    async def go():
        server, port = await serve(engine)
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(SynthesizeStart().event())
            for text in ["Увімкнено сві", "тло. У віт"]:
                await client.write_event(SynthesizeChunk(text=text).event())
            for _ in range(100):
                if engine.calls:
                    break
                await asyncio.sleep(0.02)
            calls_before_stop = [c[0] for c in engine.calls]
            for text in ["альні 2", "1 °C."]:
                await client.write_event(SynthesizeChunk(text=text).event())
            await client.write_event(Synthesize(text="Увімкнено світло. У вітальні 21 °C.").event())
            await asyncio.sleep(0.1)
            calls_before_stop2 = len(engine.calls)
            await client.write_event(SynthesizeStop().event())
            events = await read_until(client, {SynthesizeStopped().event().type})
        server.close()
        return calls_before_stop, calls_before_stop2, events

    early, before_stop, events = run(go())
    assert early == ["Увімкнено світло."]
    assert before_stop == 1
    assert [c[0] for c in engine.calls] == ["Увімкнено світло.", "У вітальні 21 °C."]
    types = [e.type for e in events]
    assert types[0] == "audio-start" and types[-2:] == ["audio-stop", "synthesize-stopped"]
    assert types.count("audio-start") == 1


def test_nothing_to_say_still_frames_audio():
    engine = FakeEngine()

    async def go():
        server, port = await serve(engine)
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="…").event())
            events = await read_until(client, {"audio-stop"})
        server.close()
        return events

    assert [e.type for e in run(go())] == ["audio-start", "audio-stop"]
    assert engine.calls == []


def test_unknown_voice_falls_back_to_default():
    engine = FakeEngine()

    async def go():
        server, port = await serve(engine)
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="Привіт.", voice=SynthesizeVoice(name="nope")).event())
            await read_until(client, {"audio-stop"})
            await client.write_event(Synthesize(text="Привіт.", voice=SynthesizeVoice(name="Speaker_1")).event())
            await read_until(client, {"audio-stop"})
        server.close()

    run(go())
    assert [c[1] for c in engine.calls] == ["Speaker_0", "Speaker_1"]


def test_concurrent_clients_are_serialized():
    engine = FakeEngine(delay=0.2)

    async def one(port):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="Привіт.").event())
            return await read_until(client, {"audio-stop"})

    async def go():
        server, port = await serve(engine)
        results = await asyncio.gather(one(port), one(port))
        server.close()
        return results

    results = run(go())
    assert engine.max_active == 1
    assert all(r[-1].type == "audio-stop" and any(e.type == "audio-chunk" for e in r) for r in results)


def test_streaming_speaks_a_finished_sentence_without_waiting():
    engine = FakeEngine()

    async def go():
        server, port = await serve(engine)
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(SynthesizeStart().event())
            await client.write_event(SynthesizeChunk(text="Увімкнено світло. ").event())
            for _ in range(100):
                if engine.calls:
                    break
                await asyncio.sleep(0.02)
            early = [c[0] for c in engine.calls]
            await client.write_event(SynthesizeChunk(text="Готово.").event())
            await client.write_event(SynthesizeStop().event())
            await read_until(client, {SynthesizeStopped().event().type})
        server.close()
        return early

    assert run(go()) == ["Увімкнено світло."]
    assert [c[0] for c in engine.calls] == ["Увімкнено світло.", "Готово."]


def hang_up_then_synthesize(engine):
    """Client A hangs up after the first audio chunk; client B then asks for speech."""

    async def go():
        server, port = await serve(engine)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await async_write_event(Synthesize(text="Привіт.").event(), writer)
        await async_read_event(reader)  # audio-start
        await async_read_event(reader)  # first chunk
        writer.transport.abort()
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="Привіт.").event())
            events = await read_until(client, {"audio-stop"})
        await asyncio.sleep(0.1)
        server.close()
        return events

    return run(go())


def test_hang_up_stops_synthesis_before_the_next_request():
    engine = FakeEngine(delay=0.05, chunks=20)
    events = hang_up_then_synthesize(engine)
    assert engine.max_active == 1
    assert engine.yielded < 30  # A stopped early; B made all 20
    assert sum(e.type == "audio-chunk" for e in events) == 20


def test_hang_up_is_not_logged_as_an_error(caplog):
    hang_up_then_synthesize(FakeEngine(delay=0.05, chunks=20))
    assert [r.getMessage() for r in caplog.records if r.levelno >= 30 or r.exc_info] == []
