"""Calibration and uncertainty helpers with explicit statistical assumptions."""

from .conformal import error_ratios, scale_error_bounds, split_conformal_quantile
from .inference import binomial_failure_summary, cluster_bootstrap_mean
from .tolerance import ToleranceCalibration, pac_tolerance_rank, pac_tolerance_threshold

__all__ = [
    "binomial_failure_summary",
    "cluster_bootstrap_mean",
    "error_ratios",
    "scale_error_bounds",
    "split_conformal_quantile",
    "ToleranceCalibration",
    "pac_tolerance_rank",
    "pac_tolerance_threshold",
]
