from ..pipeline import StepConfig, TextStep, register
from .stress import ACUTE, WORD


@register("overrides")
def overrides(config: StepConfig) -> TextStep:
    table = {o.strip().replace("+", "").lower(): o.strip().replace("+", ACUTE) for o in config.overrides if o.strip()}

    def replace(m):
        word = m.group(0)
        stressed = table.get(word.lower())
        if stressed is None:
            return word
        return stressed[0].upper() + stressed[1:] if word[0].isupper() else stressed

    def step(text: str) -> str:
        text = text.replace("+", ACUTE)
        return WORD.sub(replace, text) if table else text

    return step
