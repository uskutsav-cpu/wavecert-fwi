from .base import WavefieldSurrogate
from .low_fidelity import SmoothedHelmholtzSurrogate

__all__ = ["WavefieldSurrogate", "SmoothedHelmholtzSurrogate"]

try:
    from wavecert.surrogates.fno import FNO2dWavefield, FNOConfig, TrainedFNOWavefieldSurrogate
except ImportError:
    pass
