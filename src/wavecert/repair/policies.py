from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class BlockCertificate:
    label: str
    surrogate_derivative: float
    error_bound: float
    exact_evaluator: Callable[[], float]

    @property
    def upper_bound(self) -> float:
        return self.surrogate_derivative + self.error_bound


@dataclass(frozen=True)
class SelectiveVerificationResult:
    certified_descent: bool
    repaired_blocks: tuple[str, ...]
    final_upper_derivative: float
    exact_evaluations: int


def selectively_verify(blocks: Iterable[BlockCertificate]) -> SelectiveVerificationResult:
    """Evaluate exact blocks in decreasing uncertainty until descent is certified.

    The proposed model-space direction remains fixed.  Unverified blocks use
    ``d_hat + eta`` as a rigorous upper bound; verified blocks use their exact
    directional derivative.  This provides a clean prototype of adaptive
    shot/frequency fallback without changing the direction mid-certificate.
    """

    ordered = sorted(tuple(blocks), key=lambda b: b.error_bound, reverse=True)
    current = {b.label: b.upper_bound for b in ordered}
    repaired: list[str] = []

    total_upper = float(sum(current.values()))
    if total_upper < 0:
        return SelectiveVerificationResult(True, (), total_upper, 0)

    for block in ordered:
        exact = float(block.exact_evaluator())
        current[block.label] = exact
        repaired.append(block.label)
        total_upper = float(sum(current.values()))
        if total_upper < 0:
            return SelectiveVerificationResult(True, tuple(repaired), total_upper, len(repaired))

    return SelectiveVerificationResult(False, tuple(repaired), total_upper, len(repaired))
