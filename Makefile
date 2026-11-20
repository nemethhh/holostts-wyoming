MODEL_DIR ?= models
export HOLOSTTS_MODELS ?= $(CURDIR)/$(MODEL_DIR)
# pinned upstream revisions; the Dockerfiles use the same values
HOLOS_REV := 5d40fbaa99e55ebd3359a161bc86728ffdbd5c5f
STRESS_REV := d5b37ea0abe9711930d9c3b0b13edec5a0675512
HF_URL := https://huggingface.co/patriotyk/HolosTTS/resolve/$(HOLOS_REV)
TRIE_URL := https://raw.githubusercontent.com/patriotyk/ukrainian-word-stress/$(STRESS_REV)/ukrainian_word_stress/data/stress.trie
BUNDLE_FILES := holos_onnx.json voices.npz encoder.onnx prosody.onnx source.onnx decoder_window.onnx

.PHONY: models models-fp32 test image image-cuda image-cuda13 image-vulkan smoke

models: $(addprefix $(MODEL_DIR)/onnx-int8/,$(BUNDLE_FILES)) $(MODEL_DIR)/stress.trie
models-fp32: $(addprefix $(MODEL_DIR)/onnx/,$(BUNDLE_FILES)) $(MODEL_DIR)/stress.trie

$(MODEL_DIR)/onnx-int8/%:
	mkdir -p $(dir $@)
	curl -fsSL -o $@.tmp $(HF_URL)/onnx-int8/$* && mv $@.tmp $@

$(MODEL_DIR)/onnx/%:
	mkdir -p $(dir $@)
	curl -fsSL -o $@.tmp $(HF_URL)/onnx/$* && mv $@.tmp $@

$(MODEL_DIR)/stress.trie:
	mkdir -p $(MODEL_DIR)
	curl -fsSL -o $@.tmp $(TRIE_URL) && mv $@.tmp $@

test: models
	uv run --extra webgpu pytest

image:
	docker build -f holostts/Dockerfile --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie-2026.11.0 -t holostts-addon .

image-cuda:
	docker build -f docker/Dockerfile.cuda -t holostts-wyoming .

image-cuda13:
	docker build -f docker/Dockerfile.cuda --build-arg CUDA_BASE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 \
		--build-arg ORT_GPU=1.32.0 -t holostts-wyoming:cuda13 .

image-vulkan:
	docker build -f docker/Dockerfile.vulkan -t holostts-wyoming:vulkan .

# run IMAGE's server on 127.0.0.1:10299 and synthesize one request
SMOKE_IMAGE ?= holostts-addon
SMOKE_ARGS ?= --model-dir /models/onnx-int8
SMOKE_DOCKER_ARGS ?=
smoke:
	docker rm -f holostts-smoke >/dev/null 2>&1 || true
	docker run -d --name holostts-smoke $(SMOKE_DOCKER_ARGS) -p 127.0.0.1:10299:10200 --entrypoint /opt/venv/bin/holostts-wyoming $(SMOKE_IMAGE) $(SMOKE_ARGS)
	until docker logs holostts-smoke 2>&1 | grep -q Ready; do sleep 1; docker inspect -f '{{.State.Running}}' holostts-smoke | grep -q true || { docker logs holostts-smoke; exit 1; }; done
	uv run python tools/wyoming_client.py 127.0.0.1 10299 "Увімкнено світло у вітальні. У вітальні 21 °C." .cache/smoke.wav
	docker stats --no-stream --format '{{.MemUsage}}' holostts-smoke
	docker logs holostts-smoke 2>&1 | tail -3
	docker rm -f holostts-smoke >/dev/null
