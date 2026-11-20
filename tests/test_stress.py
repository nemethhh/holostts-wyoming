from conftest import TRIE, needs_models

from holostts_wyoming.pipeline import StepConfig, build_chain

A = "́"


@needs_models
def test_dictionary_stress():
    s = build_chain(["stress"], StepConfig(trie_path=TRIE))
    assert s("Увімкнено світло у вітальні.") == f"Уві{A}мкнено сві{A}тло у віта{A}льні."


@needs_models
def test_one_vowel_word_unchanged():
    s = build_chain(["stress"], StepConfig(trie_path=TRIE))
    assert s("П'ять хвилин") == f"П'ять хвили{A}н"


@needs_models
def test_ambiguous_word_gets_one_stress():
    s = build_chain(["stress"], StepConfig(trie_path=TRIE))
    assert s("замок") in (f"за{A}мок", f"замо{A}к")


@needs_models
def test_already_stressed_and_latin_untouched():
    s = build_chain(["stress"], StepConfig(trie_path=TRIE))
    assert s(f"сві{A}тло OK") == f"сві{A}тло OK"


@needs_models
def test_override_wins_over_dictionary():
    o = build_chain(["overrides", "stress"], StepConfig(overrides=["замо+к"], trie_path=TRIE))
    assert o("Замок відчинено") == f"Замо{A}к відчи{A}нено"


def test_plus_in_text_becomes_stress():
    assert build_chain(["overrides"], StepConfig())("Му+дрого") == f"Му{A}дрого"
