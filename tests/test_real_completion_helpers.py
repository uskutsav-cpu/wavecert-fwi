from __future__ import annotations

import numpy as np

from wavecert.experiments.real_wavecert import _conformal_quantile


def test_conformal_quantile_is_finite_and_monotone() -> None:
    scores = np.arange(1.0, 21.0)
    q90 = _conformal_quantile(scores, 0.10)
    q95 = _conformal_quantile(scores, 0.05)
    assert np.isfinite(q90)
    assert np.isfinite(q95)
    assert q95 >= q90
