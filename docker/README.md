# GPU Docker images

Two images run HolosTTS (FP32 ONNX model) on a GPU and fall back to the CPU, with a log line, when no GPU is visible:

| image | GPUs | compose file |
|---|---|---|
| `ghcr.io/nemethhh/holostts-wyoming:latest` | NVIDIA (CUDA 12), GTX 10xx and newer | [`compose.yaml`](compose.yaml) |
| `ghcr.io/nemethhh/holostts-wyoming:latest-vulkan` | AMD and Intel (Vulkan, ONNX Runtime's WebGPU EP) | [`compose.vulkan.yaml`](compose.vulkan.yaml) |

## Requirements

- **NVIDIA:** a driver that supports CUDA 12 (R525 or newer; GTX 10xx cards are supported up to the R580 driver
  branch) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- **AMD / Intel:** the kernel's `amdgpu` / `i915`/`xe` driver (no ROCm needed) and access to `/dev/dri`. The image
  brings Mesa's Vulkan drivers. On a Radeon RX 7600M XT it is about twice as fast as the CPU of a Ryzen 9 7945HX
  (first audio ~0.1 s). ROCm is not used: its ONNX Runtime support ended with ORT 1.22 and was slower than the CPU
  for this model.

## Run

```bash
mkdir holostts && cd holostts
curl -fsSLO https://raw.githubusercontent.com/nemethhh/holostts-wyoming/main/docker/compose.yaml          # NVIDIA
# curl -fsSLO https://raw.githubusercontent.com/nemethhh/holostts-wyoming/main/docker/compose.vulkan.yaml # AMD/Intel
docker compose -f compose.yaml up -d                 # AMD/Intel: RENDER_GID=$(getent group render | cut -d: -f3) docker compose -f compose.vulkan.yaml up -d
docker compose -f compose.yaml logs   # "... not available, using CPU" means the GPU is not visible to the container
```

Then in Home Assistant: it is discovered automatically (zeroconf, `network_mode: host`), or Settings → Devices &
services → Add integration → **Wyoming Protocol** → this machine's address, port `10200`.

## Settings

| environment | default | |
|---|---|---|
| `HOLOSTTS_VOICE` | `Speaker_0` | Default voice (27 presets; Home Assistant can choose per assistant). |
| `HOLOSTTS_SPEED` | 1.0 | Above 1 faster, below 1 slower. |
| `HOLOSTTS_THREADS` | 4 | CPU threads (CPU fallback and CPU-side work). |
| `HOLOSTTS_DEVICE` | `cuda` / `webgpu` | `cpu` to force the CPU. |
| `HOLOSTTS_ZEROCONF` | 1 | Announce the service for Home Assistant discovery. |
| `HOLOSTTS_DEBUG` | 0 | `1` logs each reply's text and phonemes. |

Stress fixes: one word per line, `+` after the stressed vowel (`замо+к`), in `./config/stress_overrides.txt`;
restart the container after editing.
