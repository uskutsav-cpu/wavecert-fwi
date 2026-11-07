"""Calibration and uncertainty helpers with explicit statistical assumptions."""

from .conformal import error_ratios, scale_error_bounds, split_conformal_quantile
from .inference import binomial_failure_summary, cluster_bootstrap_mean

__all__ = [
    "binomial_failure_summary",
    "cluster_bootstrap_mean",
    "error_ratios",
    "scale_error_bounds",
    "split_conformal_quantile",
]
