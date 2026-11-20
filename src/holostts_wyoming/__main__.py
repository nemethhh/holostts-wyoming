"""holostts-wyoming: Ukrainian Wyoming TTS server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from wyoming.server import AsyncServer

from . import __version__
from . import steps as _steps  # noqa: F401  (registers the text steps)
from .pipeline import DEFAULT_STEPS, StepConfig, make_phonemizer
from .server import HolosHandler, Settings, make_info

_LOGGER = logging.getLogger("holostts_wyoming")
_MODELS = Path(os.environ.get("HOLOSTTS_MODELS", "/models"))
_WARM_UP = "Привіт."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="holostts-wyoming", description=__doc__)
    p.add_argument("--uri", default=env("HOLOSTTS_URI", "tcp://0.0.0.0:10200"))
    p.add_argument("--model-dir", type=Path, default=Path(env("HOLOSTTS_MODEL_DIR", _MODELS / "onnx-int8")))
    p.add_argument("--trie", type=Path, default=_MODELS / "stress.trie")
    p.add_argument("--device", choices=["cpu", "cuda"], default=env("HOLOSTTS_DEVICE", "cpu"))
    p.add_argument("--voice", default=env("HOLOSTTS_VOICE", "Speaker_0"))
    p.add_argument("--speed", type=float, default=float(env("HOLOSTTS_SPEED", "1.0")))
    p.add_argument("--threads", type=int, default=int(env("HOLOSTTS_THREADS", "2")))
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS), help="comma-separated text steps")
    p.add_argument("--overrides-file", type=Path, default=Path(env("HOLOSTTS_OVERRIDES", "/config/stress_overrides.txt")))
    p.add_argument("--zeroconf", action="store_true", default=env("HOLOSTTS_ZEROCONF", "") == "1")
    p.add_argument("--debug", action="store_true", default=env("HOLOSTTS_DEBUG", "") == "1")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


def read_overrides(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


async def serve(args: argparse.Namespace) -> None:
    from .engines.holos import HolosEngine, providers_for

    overrides = read_overrides(args.overrides_file)
    phonemizer = make_phonemizer(StepConfig(overrides=overrides, trie_path=args.trie), args.steps.split(","))
    engine = HolosEngine(args.model_dir, providers_for(args.device), args.threads)
    voices = engine.voices()
    voice = args.voice
    if voice not in voices:
        _LOGGER.warning("Unknown default voice %r, using %s", voice, voices[0])
        voice = voices[0]
    engine.warm_up(phonemizer(_WARM_UP)[0])
    _LOGGER.info("Loaded %s (%d voices, default %s, %d overrides)", args.model_dir, len(voices), voice, len(overrides))

    server = AsyncServer.from_uri(args.uri)
    if args.zeroconf:
        from wyoming.zeroconf import HomeAssistantZeroconf

        port = urlparse(args.uri).port or 10200
        await HomeAssistantZeroconf(port=port, name="holostts").register_server()
    _LOGGER.info("Ready on %s", args.uri)
    await server.run(
        partial(HolosHandler, make_info(voices), engine, phonemizer, Settings(voice=voice, speed=args.speed), asyncio.Lock())
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
