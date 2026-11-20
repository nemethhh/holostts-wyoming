import re
import unicodedata

from ..pipeline import StepConfig, TextStep, register

_APOSTROPHES = str.maketrans({c: "'" for c in "ʼ’‘`"})
_DASHES = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_QUOTES = re.compile(r"[\"«»“”„]")
_END = ".?!:…"


@register("normalize")
def normalize(config: StepConfig) -> TextStep:
    def step(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
        text = _DASHES.sub("-", text)
        text = _QUOTES.sub("", text)
        text = " ".join(text.split())
        if text and text[-1] not in _END:
            text += "."
        return text

    return step
