"""Allocation-layout metadata only; no Model, Operator.apply or field data."""
import json
import math
from pathlib import Path

import numpy as np
from devito import Grid, TimeFunction
from devito.operator.operator import Operator as BaseOperator


def forbidden(*args, **kwargs):
    raise RuntimeError("Operator.apply forbidden during layout inspection")


BaseOperator.apply = forbidden
records = []
for label, physical, nbl, spacing, saved in [
    ("coarse40_gradient_forward", (192, 96), 40, 12.5, 2001),
    ("coarse160_forward_only", (192, 96), 160, 12.5, None),
    ("fine160_forward_only", (383, 191), 160, 6.25, None),
]:
    shape = tuple(n + 2 * nbl for n in physical)
    grid = Grid(shape=shape, extent=tuple((n - 1) * spacing for n in shape), dtype=np.float64)
    wavefield = TimeFunction(name="layout_" + label, grid=grid, time_order=2, space_order=4, save=saved)
    before = getattr(wavefield, "_data", None) is None
    allocated = tuple(int(n) for n in wavefield.shape_allocated)
    after = getattr(wavefield, "_data", None) is None
    if not before or not after:
        raise RuntimeError("layout inspection unexpectedly allocated wavefield data")
    records.append(dict(label=label, physical_shape_x_z=list(physical), nbl=nbl, spacing_m=spacing,
        logical_shape=list(wavefield.shape), allocated_shape=list(allocated),
        allocated_nbytes_if_materialized=math.prod(allocated) * 8,
        data_remained_unallocated=True, saved_samples=saved, dtype="float64"))
report = dict(status="metadata_only_no_operator_apply_no_wavefield_data_allocation", records=records,
              no_BP_arrays_or_wavefields_read=True, no_Model_constructor=True)
output = Path(__file__).with_name("saved_layout_metadata.json")
with output.open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(report))
