import ctypes
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
from holos_onnx import HolosTTS

_LOGGER = logging.getLogger(__name__)
_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"


def _cuda_device_count() -> int:
    # onnxruntime-gpu lists CUDA whenever it is compiled in; without a driver and device it crashes later
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    count = ctypes.c_int(0)
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return 0
    return count.value


def providers_for(device: str) -> list[str]:
    if device == "cuda":
        if _CUDA in ort.get_available_providers() and _cuda_device_count() > 0:
            return [_CUDA, _CPU]
        _LOGGER.warning("CUDA not available, using CPU")
    return [_CPU]


class HolosEngine:
    def __init__(self, model_dir: Path, providers: list[str], threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # matches holos_onnx's default: idle spinning threads slow the next stage down
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._model = HolosTTS(str(model_dir), streaming=True, providers=providers, session_options=options)
        self.sample_rate = self._model.sample_rate

    def voices(self) -> list[str]:
        return self._model.list_voices()

    def stream(self, phonemes: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        for chunk in self._model.stream(phonemes, voice=voice, speed=speed):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1)

    def warm_up(self, phonemes: str) -> None:
        for _ in self.stream(phonemes, self.voices()[0], 1.0):
            pass
