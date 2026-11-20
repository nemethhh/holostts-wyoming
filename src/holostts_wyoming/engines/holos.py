import ctypes
import glob
import logging
import os
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


def _session_options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    # matches holos_onnx's default: idle spinning threads slow the next stage down
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def _render_node_available() -> bool:
    # without a GPU render node Dawn may pick a software Vulkan device, far slower than the CPU EP
    return any(os.access(node, os.R_OK | os.W_OK) for node in glob.glob("/dev/dri/renderD*"))


def webgpu_session_options(threads: int) -> ort.SessionOptions | None:
    """Session options bound to the first WebGPU (Vulkan) GPU, or None when there is none."""
    if not _render_node_available():
        _LOGGER.warning("WebGPU not available (no GPU render node in /dev/dri), using CPU")
        return None
    try:
        import onnxruntime_ep_webgpu as webgpu
    except ImportError:
        _LOGGER.warning("WebGPU not available (onnxruntime-ep-webgpu is not installed), using CPU")
        return None
    try:
        ort.register_execution_provider_library("webgpu", webgpu.get_library_path())
    except Exception as error:
        if "already registered" not in str(error):
            _LOGGER.warning("WebGPU not available (%s), using CPU", error)
            return None
    devices = [
        d for d in ort.get_ep_devices()
        if d.ep_name == webgpu.get_ep_name() and d.device.type == ort.OrtHardwareDeviceType.GPU
    ]
    if not devices:
        _LOGGER.warning("WebGPU not available (no GPU found), using CPU")
        return None
    options = _session_options(threads)
    options.log_severity_level = 3  # shape ops placed on CPU are expected; don't warn per session
    options.add_provider_for_devices(devices[:1], {})
    return options


class HolosEngine:
    def __init__(self, model_dir: Path, providers: list[str], threads: int):
        self._model = HolosTTS(str(model_dir), streaming=True, providers=providers, session_options=_session_options(threads))
        self.sample_rate = self._model.sample_rate
        self.device = "cuda" if _CUDA in providers else "cpu"

    def use_webgpu(self, options: ort.SessionOptions) -> None:
        """Rebuild the four stage sessions on WebGPU (holos_onnx always passes its own provider list)."""
        from .convtranspose import rewrite_conv_transpose
        import onnx

        model_dir = self._model.model_dir
        manifest, streaming = self._model.manifest, self._model.streaming
        files = {
            "encoder": manifest["encoder"],
            "prosody": manifest["prosody"],
            "source": streaming["source"],
            "decoder": streaming["decoder"],
        }
        sessions = {}
        for stage, filename in files.items():
            model = onnx.load(model_dir / filename)
            if stage == "decoder":
                model = rewrite_conv_transpose(model)
            sessions[stage] = ort.InferenceSession(model.SerializeToString(), options)
        for stage, session in sessions.items():
            setattr(self._model, f"{stage}_session", session)
        self.device = "webgpu"

    def voices(self) -> list[str]:
        return self._model.list_voices()

    def stream(self, phonemes: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        for chunk in self._model.stream(phonemes, voice=voice, speed=speed):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1)

    def warm_up(self, phonemes: str) -> None:
        for _ in self.stream(phonemes, self.voices()[0], 1.0):
            pass


def _accelerated_engine(model_dir: Path, device: str, threads: int) -> HolosEngine:
    if device != "webgpu":
        return HolosEngine(model_dir, providers_for(device), threads)
    engine = HolosEngine(model_dir, [_CPU], threads)
    options = webgpu_session_options(threads)
    if options is not None:
        try:
            engine.use_webgpu(options)
        except Exception as error:
            _LOGGER.warning("WebGPU failed (%s), using CPU", error)
    return engine


def make_engine(model_dir: Path, device: str, threads: int, warm_up: str | None = None) -> HolosEngine:
    """device: cpu, cuda or webgpu (Vulkan: AMD, Intel); falls back to CPU with a warning.

    With warm_up phonemes, one synthesis runs first, so a GPU that loads but cannot run also falls back.
    """
    engine = _accelerated_engine(model_dir, device, threads)
    if warm_up is None:
        return engine
    try:
        engine.warm_up(warm_up)
        return engine
    except Exception as error:
        if engine.device == "cpu":
            raise
        _LOGGER.warning("Synthesis on %s failed (%s), using CPU", engine.device, error)
    engine = HolosEngine(model_dir, [_CPU], threads)
    engine.warm_up(warm_up)
    return engine
