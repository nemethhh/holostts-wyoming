# GPU Docker image

`ghcr.io/nemethhh/holostts-wyoming` runs HolosTTS (FP32 ONNX model) on an NVIDIA GPU with CUDA, and falls back to
the CPU with a log line if no GPU is visible.

## Requirements

- An NVIDIA GPU from the GTX 10xx (Pascal) generation or newer, and a driver that supports CUDA 12 (R525 or newer;
  Pascal cards are supported up to the R580 driver branch).
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

## Run

```bash
mkdir holostts && cd holostts
curl -fsSLO https://raw.githubusercontent.com/nemethhh/holostts-wyoming/main/docker/compose.yaml
docker compose up -d
docker compose logs   # "CUDA not available, using CPU" means the GPU is not visible to the container
```

Then in Home Assistant: it is discovered automatically (zeroconf, `network_mode: host`), or Settings → Devices &
services → Add integration → **Wyoming Protocol** → this machine's address, port `10200`.

## Settings

| environment | default | |
|---|---|---|
| `HOLOSTTS_VOICE` | `Speaker_0` | Default voice (27 presets; Home Assistant can choose per assistant). |
| `HOLOSTTS_SPEED` | 1.0 | Above 1 faster, below 1 slower. |
| `HOLOSTTS_THREADS` | 4 | CPU threads (CPU fallback and CPU-side work). |
| `HOLOSTTS_DEVICE` | `cuda` | `cpu` to force the CPU. |
| `HOLOSTTS_ZEROCONF` | 1 | Announce the service for Home Assistant discovery. |
| `HOLOSTTS_DEBUG` | 0 | `1` logs each reply's text and phonemes. |

Stress fixes: one word per line, `+` after the stressed vowel (`замо+к`), in `./config/stress_overrides.txt`;
restart the container after editing.
