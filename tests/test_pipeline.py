import pytest

import logging

from holostts_wyoming.pipeline import StepConfig, build_chain, register, split_sentences, take_sentences


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


def test_overlong_sentence_is_split_for_the_engine():
    from holostts_wyoming.pipeline import make_phonemizer

    register("t_id")(lambda c: lambda s: s)
    sentence = ", ".join(["увімкнено світло у вітальні та на кухні"] * 12) + "."
    parts = make_phonemizer(StepConfig(), ["t_id"])(sentence)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    assert " ".join(parts).replace(",", "").split() == sentence.replace(",", "").split()


def test_overlong_sentence_without_punctuation_splits_at_spaces():
    from holostts_wyoming.pipeline import make_phonemizer

    register("t_id")(lambda c: lambda s: s)
    sentence = " ".join(["слово"] * 80)
    parts = make_phonemizer(StepConfig(), ["t_id"])(sentence)
    assert all(len(p) <= 200 for p in parts) and " ".join(parts).split() == sentence.split()


@pytest.mark.parametrize(
    "buffer,done,rest",
    [
        ("Увімкнено світло. ", ["Увімкнено світло."], ""),
        ("Привіт. Як ", ["Привіт."], "Як "),
        ("Раз. Два! Три", ["Раз.", "Два!"], "Три"),
        ("У вітальні 21", [], "У вітальні 21"),
        ("Привіт.\n", ["Привіт."], ""),
    ],
)
def test_take_sentences_from_a_stream(buffer, done, rest):
    assert take_sentences(buffer) == (done, rest)


def test_chain_logs_each_step_at_debug(caplog):
    register("t_up")(lambda c: lambda s: s.upper())
    register("t_x")(lambda c: lambda s: s + "!")
    with caplog.at_level(logging.DEBUG, logger="holostts_wyoming.pipeline"):
        build_chain(["t_up", "t_x"], StepConfig())("ok")
    assert [r.getMessage() for r in caplog.records] == ["t_up: OK", "t_x: OK!"]
