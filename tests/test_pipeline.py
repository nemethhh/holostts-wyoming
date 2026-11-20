import pytest

from holostts_wyoming.pipeline import StepConfig, build_chain, register, split_sentences


def test_split_sentences_on_punctuation_and_newlines():
    assert split_sentences("Привіт. Як справи?\nДобре!") == ["Привіт.", "Як справи?", "Добре!"]


def test_split_sentences_keeps_decimal_numbers():
    assert split_sentences("Температура 21.5 градуса") == ["Температура 21.5 градуса"]


def test_split_sentences_on_ellipsis():
    assert split_sentences("Ну… добре") == ["Ну…", "добре"]


def test_split_sentences_blank():
    assert split_sentences("  ") == []


def test_build_chain_runs_steps_in_order():
    register("t_a")(lambda c: lambda s: s + "a")
    register("t_b")(lambda c: lambda s: s + "b")
    assert build_chain(["t_a", "t_b"], StepConfig())("") == "ab"


def test_build_chain_unknown_step_lists_known():
    register("t_a")(lambda c: lambda s: s + "a")
    with pytest.raises(ValueError, match="t_a"):
        build_chain(["nope"], StepConfig())
