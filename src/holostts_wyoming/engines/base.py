from typing import Iterator, Protocol

import numpy as np


class Engine(Protocol):
    sample_rate: int

    def voices(self) -> list[str]: ...

    def stream(self, phonemes: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        """Float32 mono chunks for one sentence of phonemes."""
        ...
