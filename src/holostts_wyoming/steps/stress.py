import os
import re
from pathlib import Path

from ..pipeline import StepConfig, TextStep, register
from ..stressdict import StressDict

ACUTE = "́"
WORD = re.compile(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ'́]+")


def default_trie() -> Path:
    return Path(os.environ.get("HOLOSTTS_MODELS", "/models")) / "stress.trie"


@register("stress")
def stress(config: StepConfig) -> TextStep:
    dictionary = StressDict(config.trie_path or default_trie())

    def place(m: re.Match) -> str:
        word = m.group(0)
        if ACUTE in word:
            return word
        for pos in sorted(dictionary.accent_positions(word), reverse=True):
            word = word[:pos] + ACUTE + word[pos:]
        return word

    def step(text: str) -> str:
        return WORD.sub(place, text)

    return step
