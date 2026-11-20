from pathlib import Path

from holostts_wyoming.__main__ import parse_args, read_overrides
from holostts_wyoming.pipeline import DEFAULT_STEPS


def test_defaults():
    args = parse_args([])
    assert args.uri == "tcp://0.0.0.0:10200"
    assert args.device == "cpu"
    assert args.voice == "Speaker_0"
    assert args.speed == 1.0
    assert args.threads == 2
    assert args.steps == ",".join(DEFAULT_STEPS)
    assert args.zeroconf is False


def test_overrides_file(tmp_path: Path):
    f = tmp_path / "o.txt"
    f.write_text("# my fixes\nзамо+к\n\n  Му+дрого  \n", encoding="utf-8")
    assert read_overrides(f) == ["замо+к", "Му+дрого"]


def test_missing_overrides_file_is_empty(tmp_path: Path):
    assert read_overrides(tmp_path / "none.txt") == []


def test_webgpu_device_accepted():
    assert parse_args(["--device", "webgpu"]).device == "webgpu"
