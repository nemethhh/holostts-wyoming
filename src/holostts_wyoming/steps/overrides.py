import re

from ..pipeline import StepConfig, TextStep, register
from .normalize import normalize_characters
from .stress import ACUTE, WORD


_PLUS_AFTER_VOWEL = re.compile(r"(?<=[АЕЄИІЇОУЮЯаеєиіїоуюя])\+")


@register("overrides")
def overrides(config: StepConfig) -> TextStep:
    entries = [normalize_characters(o.strip()).replace("+", ACUTE) for o in config.overrides if o.strip()]
    table = {e.replace(ACUTE, "").lower(): e for e in entries}

    def replace(m):
        word = m.group(0)
        stressed = table.get(word.lower())
        if stressed is None:
            return word
        return stressed[0].upper() + stressed[1:] if word[0].isupper() else stressed

    def step(text: str) -> str:
        text = _PLUS_AFTER_VOWEL.sub(ACUTE, text)
        return WORD.sub(replace, text) if table else text

    return step
