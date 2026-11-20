import json

from conftest import INT8, TRIE, needs_models

from holostts_wyoming.pipeline import StepConfig, phonemize

cfg = StepConfig(trie_path=TRIE)


def vocab():
    return set(json.load(open(INT8 / "holos_onnx.json"))["vocab"])


@needs_models
def test_two_sentences_in_vocab_with_stress():
    out = phonemize("Увімкнено світло у вітальні. У вітальні 21 °C!", cfg)
    assert len(out) == 2
    assert all(set(p) <= vocab() for p in out)
    assert "ˈ" in out[0]


@needs_models
def test_latin_only_text_is_safe():
    out = phonemize("OK", cfg)
    assert out == [] or set(out[0]) <= vocab()


@needs_models
def test_punctuation_only_is_dropped():
    assert phonemize("…", cfg) == []
    assert phonemize("", cfg) == []
