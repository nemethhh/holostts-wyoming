"""Synthesize text with a Wyoming TTS server into a WAV: wyoming_client.py HOST PORT TEXT OUT.wav [--voice V] [--stream]"""

import argparse
import asyncio
import time
import wave

from wyoming.audio import AudioChunk, AudioStart
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("host")
    p.add_argument("port", type=int)
    p.add_argument("text")
    p.add_argument("out")
    p.add_argument("--voice")
    p.add_argument("--stream", action="store_true", help="send the text as streamed chunks of 10 characters")
    a = p.parse_args()
    voice = SynthesizeVoice(name=a.voice) if a.voice else None
    started = time.monotonic()
    first = None
    audio, rate = bytearray(), 24000
    async with AsyncTcpClient(a.host, a.port) as client:
        if a.stream:
            await client.write_event(SynthesizeStart(voice=voice).event())
            for i in range(0, len(a.text), 10):
                await client.write_event(SynthesizeChunk(text=a.text[i : i + 10]).event())
            await client.write_event(SynthesizeStop().event())
            end = "synthesize-stopped"
        else:
            await client.write_event(Synthesize(text=a.text, voice=voice).event())
            end = "audio-stop"
        while (event := await client.read_event()) is not None and event.type != end:
            if AudioStart.is_type(event.type):
                rate = AudioStart.from_event(event).rate
            elif AudioChunk.is_type(event.type):
                first = first or time.monotonic()
                audio += AudioChunk.from_event(event).audio
            elif event.type == "error":
                raise SystemExit(f"server error: {event.data}")
    total = time.monotonic() - started
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(audio)
    seconds = len(audio) / 2 / rate
    print(f"first audio {((first or started) - started) * 1000:.0f} ms, total {total * 1000:.0f} ms, {seconds:.2f} s of audio, RTF {total / max(seconds, 1e-9):.3f}")


asyncio.run(main())
