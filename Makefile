MODEL_DIR ?= models
export HOLOSTTS_MODELS ?= $(CURDIR)/$(MODEL_DIR)
# pinned upstream revisions; the Dockerfiles use the same values
HOLOS_REV := 5d40fbaa99e55ebd3359a161bc86728ffdbd5c5f
STRESS_REV := d5b37ea0abe9711930d9c3b0b13edec5a0675512
HF_URL := https://huggingface.co/patriotyk/HolosTTS/resolve/$(HOLOS_REV)
TRIE_URL := https://raw.githubusercontent.com/patriotyk/ukrainian-word-stress/$(STRESS_REV)/ukrainian_word_stress/data/stress.trie
BUNDLE_FILES := holos_onnx.json voices.npz encoder.onnx prosody.onnx source.onnx decoder_window.onnx

.PHONY: models models-fp32 test image image-cuda smoke

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
	uv run pytest
