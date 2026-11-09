"""Offline calibration scores for a complete, fixed-state greedy repair path."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.statistics.conformal import error_ratios
from wavecert.surrogates.base import WavefieldSurrogate


@dataclass(frozen=True)
class PathwiseScore:
    score: float
    certificates_scored: int
    repaired_order: tuple[str, ...]


def pathwise_case_score_details(
    *,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m: np.ndarray,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
) -> PathwiseScore:
    """Score the largest error/bound ratio over all blocks and repair rounds.

    The exact gradients are evaluated solely to construct offline calibration
    labels. Deployment must never use these validation gradients. The greedy
    order uses deterministic, unscaled bounds with input-order tie-breaking.
    A *strictly positive finite uniform* multiplier preserves that order, so
    the maximum covers any stopping round along this one fixed-state path.

    This does not cover future inversion iterates or distribution shift. It
    does not cover a policy using separate block scales or a different repair
    order. Independent/exchangeable calibration cases are a caller obligation.
    """

    m = physics.validate_model(m)
    blocks = tuple(blocks)
    if not blocks or len({block.label for block in blocks}) != len(blocks):
        raise ValueError("pathwise calibration requires nonempty, uniquely labelled blocks")
    states = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, ng, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        _, eg, _, _ = physics.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        states.append({"block": block, "q": q, "neural": ng, "validation_exact": eg, "repaired": False})
    frequencies = sorted({float(block.frequency_hz) for block in blocks})
    constants = {}
    for frequency in frequencies:
        matrix = physics.receiver_resolvent_matrix(m, frequency, receiver_indices)
        alpha = physics.receiver_resolvent_norm(
            m, frequency, receiver_indices, receiver_resolvent_matrix=matrix
        )
        constants[frequency] = (matrix, alpha)
    worst = 0.0
    certificates = 0
    repaired_order = []
    while True:
        gradient = np.sum([
            state["validation_exact"] if state["repaired"] else state["neural"]
            for state in states
        ], axis=0)
        norm = float(np.linalg.norm(gradient))
        unrepaired = [state for state in states if not state["repaired"]]
        if norm <= 1e-14 or not unrepaired:
            break
        direction = -gradient / norm
        kappas = {}
        for frequency in {float(state["block"].frequency_hz) for state in unrepaired}:
            matrix, _ = constants[frequency]
            kappas[frequency] = physics.directional_receiver_tangent_resolvent_norm(
                m, frequency, receiver_indices, direction, receiver_resolvent_matrix=matrix
            )
        uncertainty = []
        for state in unrepaired:
            block = state["block"]
            frequency = float(block.frequency_hz)
            matrix, alpha = constants[frequency]
            cert = certify_direction(
                physics=physics, surrogate=surrogate, m=m, q=state["q"],
                observed=block.observed, receiver_indices=receiver_indices,
                frequency_hz=frequency, direction=direction, stability_mode="directional",
                receiver_resolvent_matrix=matrix, receiver_resolvent_norm=alpha,
                directional_tangent_resolvent_norm=kappas[frequency],
            )
            exact_dd = float(np.dot(state["validation_exact"], direction))
            error = abs(exact_dd - cert.surrogate_directional_derivative)
            ratio = float(error_ratios(np.asarray(error), np.asarray(cert.directional_error_bound)))
            worst = max(worst, ratio)
            certificates += 1
            uncertainty.append((state, cert.directional_error_bound))
        selected = max(uncertainty, key=lambda item: item[1])[0]
        selected["repaired"] = True
        repaired_order.append(selected["block"].label)
    return PathwiseScore(float(worst), certificates, tuple(repaired_order))


def pathwise_case_score(**kwargs) -> float:
    """Return only the maximum score; see :func:`pathwise_case_score_details`."""

    return pathwise_case_score_details(**kwargs).score
