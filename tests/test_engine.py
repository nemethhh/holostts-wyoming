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


def test_cuda_falls_back_to_cpu_when_unavailable(caplog):
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return
    assert providers_for("cuda") == ["CPUExecutionProvider"]
    assert "CUDA not available" in caplog.text
