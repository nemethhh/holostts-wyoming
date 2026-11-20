"""Numbers, decimals, times, degrees and percentages to Ukrainian words."""

from __future__ import annotations

import re
from functools import cache

import pymorphy3
from num2words import num2words

from ..pipeline import StepConfig, TextStep, register

# nominative singular, nominative plural, genitive plural, genitive singular
_SYMBOL_NOUNS = {
    "°C": ("градус", "градуси", "градусів", "градуса"),
    "°F": ("градус", "градуси", "градусів", "градуса"),
    "°": ("градус", "градуси", "градусів", "градуса"),
    "%": ("відсоток", "відсотки", "відсотків", "відсотка"),
}
_WORD = r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ'́]+"

_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_NUMBER = re.compile(
    r"(?<![\w.,])([+-])?(\d+)(?:[.,](\d+))?(?![\d])"
    r"(?:\s*(°C|°F|°|%)|(\s+)(" + _WORD + r"))?"
)
_DIGITS = re.compile(r"\d+")
# hours as feminine ordinals, stressed here: the dictionary's first record for п'ята / сьома is another word
_A = "\u0301"
_HOURS = [
    "нуль", f"пе{_A}рша", f"дру{_A}га", f"тре{_A}тя", f"четве{_A}рта", f"п'я{_A}та", f"шо{_A}ста", f"сьо{_A}ма",
    f"во{_A}сьма", f"дев'я{_A}та", f"деся{_A}та", f"одина{_A}дцята", f"двана{_A}дцята", f"трина{_A}дцята",
    f"чотирна{_A}дцята", f"п'ятна{_A}дцята", f"шістна{_A}дцята", f"сімна{_A}дцята", f"вісімна{_A}дцята",
    f"дев'ятна{_A}дцята", f"двадця{_A}та", f"два{_A}дцять пе{_A}рша", f"два{_A}дцять дру{_A}га", f"два{_A}дцять тре{_A}тя",
]


def _words(n: int, feminine: bool = False, case: str | None = None) -> str:
    kwargs = {"lang": "uk", "gender": "feminine" if feminine else "masculine"}
    if case:
        kwargs["case"] = case
    return num2words(n, **kwargs)


@cache
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer(lang="uk")


def _agreement(noun: str) -> tuple[str | None, bool]:
    """The noun's gender (masc, femn, neut or None) and whether it is accusative singular."""
    for parse in _morph().parse(noun.replace("\u0301", "").lower()):
        if "NOUN" in parse.tag:
            return parse.normalized.tag.gender, "accs" in parse.tag and "plur" not in parse.tag
    return None, False


def _count_form(n: int) -> int:
    if 11 <= n % 100 <= 14:
        return 2
    if n % 10 == 1:
        return 0
    if 2 <= n % 10 <= 4:
        return 1
    return 2


def _decimal(whole: int, frac: str) -> str:
    whole_part = "ціла" if whole % 10 == 1 and whole % 100 != 11 else "цілих"
    f = int(frac)
    one = f % 10 == 1 and f % 100 != 11
    denominators = {1: ("десята", "десятих"), 2: ("сота", "сотих"), 3: ("тисячна", "тисячних")}
    denominator = denominators.get(len(frac), denominators[3])[0 if one else 1]
    return f"{_words(whole, feminine=True)} {whole_part} {_words(f, feminine=True)} {denominator}"


def _time(m: re.Match) -> str:
    hour, minutes = int(m.group(1)), m.group(2)
    if minutes == "00":
        tail = "нуль нуль"
    elif minutes.startswith("0"):
        tail = "нуль " + _words(int(minutes), feminine=True)
    else:
        tail = _words(int(minutes), feminine=True)
    return f"{_HOURS[hour]} {tail}"


def _number(m: re.Match) -> str:
    sign, whole, frac, symbol, space, noun = m.groups()
    n = int(whole)
    prefix = {"-": "мінус ", "+": "плюс "}.get(sign or "", "")
    gender, accusative = _agreement(noun) if noun else (None, False)
    feminine = gender == "femn"
    if frac is not None:
        spoken = _decimal(n, frac)
    else:
        spoken = _words(n, feminine=feminine, case="accusative" if feminine and accusative else None)
        if gender == "neut" and spoken.endswith("один"):
            spoken = spoken[: -len("один")] + "одне"  # num2words has no neuter for Ukrainian

    if symbol:
        forms = _SYMBOL_NOUNS[symbol]
        unit = forms[3] if frac is not None else forms[_count_form(n)]
        return f"{prefix}{spoken} {unit}"
    if noun:
        return f"{prefix}{spoken}{space}{noun}"
    return prefix + spoken


@register("numbers")
def numbers(config: StepConfig) -> TextStep:
    def step(text: str) -> str:
        text = _TIME.sub(_time, text)
        text = _NUMBER.sub(_number, text)
        # whatever the rules above left (versions, addresses, "1 000") is read digit group by digit group
        return _DIGITS.sub(lambda m: _words(int(m.group(0))), text)

    return step
