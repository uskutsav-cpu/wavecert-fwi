from __future__ import annotations

import importlib.util
import platform
from dataclasses import asdict

from wavecert.physics.devito_backend import DevitoAcousticConfig


def devito_environment_report() -> dict:
    """Describe whether the optional Devito production backend can execute."""

    available = importlib.util.find_spec("devito") is not None
    seismic_examples = False
    if available:
        try:
            seismic_examples = importlib.util.find_spec("examples.seismic") is not None
        except ModuleNotFoundError:
            seismic_examples = False
    return {
        "devito_installed": available,
        "devito_seismic_examples_available": seismic_examples,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reference_config": asdict(DevitoAcousticConfig()),
        "executable": bool(available and seismic_examples),
    }
