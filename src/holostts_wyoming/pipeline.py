"""Text front end: a chain of named str -> str steps, run per sentence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

TextStep = Callable[[str], str]
StepFactory = Callable[["StepConfig"], TextStep]
F = TypeVar("F", bound=StepFactory)

DEFAULT_STEPS = ["normalize", "numbers", "overrides", "stress", "ipa"]

_REGISTRY: dict[str, StepFactory] = {}
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass
class StepConfig:
    overrides: list[str] = field(default_factory=list)
    trie_path: Path | None = None


def register(name: str) -> Callable[[F], F]:
    def decorator(factory: F) -> F:
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_chain(names: list[str], config: StepConfig) -> TextStep:
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"unknown steps {unknown}; known: {sorted(_REGISTRY)}")
    steps = [_REGISTRY[n](config) for n in names]

    def chain(text: str) -> str:
        for step in steps:
            text = step(text)
        return text

    return chain


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
