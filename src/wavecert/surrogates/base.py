from __future__ import annotations

from typing import Protocol

import numpy as np

Array = np.ndarray


class WavefieldSurrogate(Protocol):
    """Minimal interface required by the a-posteriori certifier."""

    def state(self, m: Array, q: Array, frequency_hz: float) -> Array: ...

    def jvp(self, m: Array, direction: Array, q: Array, frequency_hz: float) -> Array: ...

    def objective_and_gradient(
        self,
        m: Array,
        q: Array,
        observed: Array,
        receiver_indices: tuple[int, ...],
        frequency_hz: float,
    ) -> tuple[float, Array, Array, Array]: ...
