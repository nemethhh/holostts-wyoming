import os
from pathlib import Path

import pytest

import holostts_wyoming.steps  # noqa: F401  (registers the steps)

MODELS = Path(os.environ.get("HOLOSTTS_MODELS", Path(__file__).parent.parent / "models"))
TRIE = MODELS / "stress.trie"
INT8 = MODELS / "onnx-int8"

needs_models = pytest.mark.skipif(not (INT8 / "holos_onnx.json").is_file() or not TRIE.is_file(), reason="run `make models`")
