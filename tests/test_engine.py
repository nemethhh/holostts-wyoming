import numpy as np
from conftest import INT8, TRIE, needs_models

from holostts_wyoming.engines.holos import HolosEngine, providers_for
from holostts_wyoming.pipeline import StepConfig, phonemize


@needs_models
def test_holos_engine_streams_speech():
    engine = HolosEngine(INT8, providers_for("cpu"), threads=2)
    assert len(engine.voices()) == 27 and "Speaker_0" in engine.voices()
    [phonemes] = phonemize("Увімкнено світло у вітальні.", StepConfig(trie_path=TRIE))
    chunks = list(engine.stream(phonemes, "Speaker_0", 1.0))
    audio = np.concatenate(chunks)
    assert len(chunks) >= 1
    assert audio.size > 0.5 * engine.sample_rate
    assert np.isfinite(audio).all()


def test_cuda_compiled_in_but_no_device_falls_back_to_cpu(monkeypatch, caplog):
    import holostts_wyoming.engines.holos as holos

    monkeypatch.setattr(holos.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(holos, "_cuda_device_count", lambda: 0)
    assert providers_for("cuda") == ["CPUExecutionProvider"]
    assert "CUDA not available" in caplog.text


def test_cuda_with_device(monkeypatch):
    import holostts_wyoming.engines.holos as holos

    monkeypatch.setattr(holos.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(holos, "_cuda_device_count", lambda: 1)
    assert providers_for("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_device_count_without_driver_is_zero():
    import ctypes.util

    from holostts_wyoming.engines.holos import _cuda_device_count

    if ctypes.util.find_library("cuda") is None:
        assert _cuda_device_count() == 0
