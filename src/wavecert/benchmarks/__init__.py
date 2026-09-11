"""Dataset adapters and compact geology stress-test generators."""

from wavecert.benchmarks.marmousi import MARMOUSI_PUBLIC_URL, load_marmousi_binary
from wavecert.benchmarks.openfwi import load_openfwi_velocity_batch
from wavecert.benchmarks.procedural import GEOLOGY_SETTINGS, procedural_geology
from wavecert.benchmarks.subsurfacegen import (
    SUBSURFACEGEN_SETTINGS,
    load_subsurfacegen_velocity_slice,
)

__all__ = [
    "GEOLOGY_SETTINGS",
    "MARMOUSI_PUBLIC_URL",
    "SUBSURFACEGEN_SETTINGS",
    "load_marmousi_binary",
    "load_openfwi_velocity_batch",
    "load_subsurfacegen_velocity_slice",
    "procedural_geology",
]
