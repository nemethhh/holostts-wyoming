import os

import pytest
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


def test_webgpu_without_plugin_falls_back_to_cpu(monkeypatch, caplog):
    import builtins

    import holostts_wyoming.engines.holos as holos

    real_import = builtins.__import__

    def no_webgpu(name, *args, **kwargs):
        if name == "onnxruntime_ep_webgpu":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_webgpu)
    assert holos.webgpu_session_options(threads=2) is None
    assert "WebGPU not available" in caplog.text


FP32 = INT8.parent / "onnx"
needs_webgpu = pytest.mark.skipif(
    os.environ.get("HOLOSTTS_TEST_WEBGPU") != "1" or not (FP32 / "decoder_window.onnx").is_file(),
    reason="set HOLOSTTS_TEST_WEBGPU=1 on a machine with a Vulkan GPU, and run `make models-fp32`",
)


@needs_webgpu
def test_webgpu_engine_matches_cpu():
    from holostts_wyoming.engines.holos import make_engine

    [phonemes] = phonemize("Увімкнено світло у вітальні.", StepConfig(trie_path=TRIE))
    gpu = make_engine(FP32, "webgpu", threads=2)
    cpu = make_engine(FP32, "cpu", threads=2)
    assert gpu.device == "webgpu" and cpu.device == "cpu"
    a = np.concatenate(list(cpu.stream(phonemes, "Speaker_0", 1.0)))
    b = np.concatenate(list(gpu.stream(phonemes, "Speaker_0", 1.0)))
    assert a.size == b.size
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_webgpu_without_render_node_falls_back_to_cpu(monkeypatch, caplog):
    import holostts_wyoming.engines.holos as holos

    monkeypatch.setattr(holos, "_render_node_available", lambda: False)
    assert holos.webgpu_session_options(threads=2) is None
    assert "no GPU" in caplog.text
