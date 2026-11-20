# Changelog

## 0.1.2

- No change to the add-on. New Docker image `holostts-wyoming:latest-cuda13` (CUDA 13, current ONNX Runtime) for RTX 20xx and newer NVIDIA cards.

## 0.1.1

- Numbers agree with any noun, not only units: "1 лампа" → "одна лампа", "1 вікно" → "одне вікно", "через 1 хвилину" → "через одну хвилину".
- Streamed replies: a finished sentence is spoken right away; a word split across text chunks is no longer glued together.
- A client that hangs up mid-reply stops its synthesis before the next request starts, and is logged in one line.
- Stress overrides match words typed with ’ or pasted with stress marks.
- Debug logging shows the text after each step.

## 0.1.0

- First release: HolosTTS INT8 on CPU, dictionary stress, numbers/times/units to words, streamed text input.
