from holostts_wyoming.pipeline import StepConfig, build_chain

n = build_chain(["normalize"], StepConfig())


def test_apostrophes_and_spaces():
    assert n("П’ять  хвилин") == "П'ять хвилин."


def test_quotes_removed_dash_kept():
    assert n("«Світло» — увімкнено!") == "Світло - увімкнено!"


def test_unicode_minus_before_digit():
    assert n("Температура −5 °C") == "Температура -5 °C."


def test_latin_only():
    assert n("OK") == "OK."


def test_empty():
    assert n("") == ""


def test_keeps_stress_marks_and_plus():
    assert n("Му+дрого сві́тло") == "Му+дрого сві́тло."
