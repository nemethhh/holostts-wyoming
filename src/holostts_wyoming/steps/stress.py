import os
import re
from pathlib import Path

from ..pipeline import StepConfig, TextStep, register
from ..stressdict import StressDict

ACUTE = "\u0301"
_LETTERS = r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ'\u0301]+"
WORD = re.compile(_LETTERS + r"(?:-" + _LETTERS + r")*")


def default_trie() -> Path:
    return Path(os.environ.get("HOLOSTTS_MODELS", "/models")) / "stress.trie"


@register("stress")
def stress(config: StepConfig) -> TextStep:
    dictionary = StressDict(config.trie_path or default_trie())

    def stressed(word: str) -> str:
        for pos in sorted(dictionary.accent_positions(word), reverse=True):
            word = word[:pos] + ACUTE + word[pos:]
        return word

    def place(m: re.Match) -> str:
        word = m.group(0)
        if ACUTE in word:
            return word
        whole = stressed(word)
        if whole != word or "-" not in word:
            return whole
        # hyphenated form missing from the dictionary: stress each part
        return "-".join(part if ACUTE in part else stressed(part) for part in word.split("-"))

    def step(text: str) -> str:
        return WORD.sub(place, text)

    return step
