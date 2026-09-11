from wavecert.surrogates.base import WavefieldSurrogate
from wavecert.surrogates.fno import FNO2dWavefield, FNOConfig, TrainedFNOWavefieldSurrogate
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate

__all__ = [
    "FNO2dWavefield",
    "FNOConfig",
    "SmoothedHelmholtzSurrogate",
    "TrainedFNOWavefieldSurrogate",
    "WavefieldSurrogate",
]
