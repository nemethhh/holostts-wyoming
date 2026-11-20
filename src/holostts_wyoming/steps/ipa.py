from ipa_uk import ipa as to_ipa

from ..pipeline import StepConfig, TextStep, register


@register("ipa")
def ipa(config: StepConfig) -> TextStep:
    return to_ipa
