"""Fresh primary C authority: no Model, no Operator.apply, no saved-u data."""
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import devito
import numpy as np
from devito import Function, Grid
from devito.operator.operator import Operator as BaseOperator
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic.operators import ForwardOperator, GradientOperator
from examples.seismic.model import PhysicalDomain

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "primary_metadata"
CACHE = Path("/var/folders/f4/8spnm17n3g9cvbpydqnw8m380000gn/T/devito-jitcache-uid501")
CACHED = {"rolling_forward": "217654e5bcbe2bcd652ce9fe3bef6e0cc12bc044.c",
          "saved_forward": "9ba19aea750045ed5f015618041722d71c732228.c",
          "gradient": "098077760fa288827113843ea63a980fe02f53f9.c"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError("Every Operator.apply is forbidden in this metadata query")


BaseOperator.apply = forbidden
if OUTPUT.exists():
    raise FileExistsError("fresh metadata output only; preserve every earlier attempt")
OUTPUT.mkdir()
records = []
for label in CACHED:
    grid = Grid(shape=(272, 176), extent=(3387.5, 2187.5), origin=(-500., -500.),
                dtype=np.float64, subdomains=(PhysicalDomain(4, fs=False),))
    vp = Function(name="vp", grid=grid, space_order=4, dtype=np.float64)
    vp.data[:] = 2.  # Own analytic constant, not a native/BP field.
    damp = Function(name="damp", grid=grid, space_order=4, dtype=np.float64)
    damp.data[:] = 0.  # Numerical damping is reconstructed separately by the port.
    model = SimpleNamespace(dim=2, grid=grid, critical_dt=.5, m=1/(vp*vp),
                            damp=damp, fs=False, spacing_map=grid.spacing_map)
    xs = np.asarray([16, 32, 48, 64, 80, 112, 128, 144, 160, 176])*12.5
    receivers = np.column_stack((xs, np.full(10, 200.)))
    geometry = AcquisitionGeometry(model, receivers, np.asarray([[1200., 200.]]),
                                   0., 1000., f0=.005, src_type="Ricker")
    geometry.resample(.5)
    if label == "gradient":
        operator = GradientOperator(model, geometry, space_order=4, save=True)
    else:
        operator = ForwardOperator(model, geometry, space_order=4, save=label == "saved_forward")
    fields = [f for f in operator.input if getattr(f, "is_TimeFunction", False) and f.name == "u"]
    assert len(fields) == 1
    if label != "rolling_forward":
        assert fields[0].shape[0] == 2001 and fields[0]._data is None
    code = str(operator.ccode).encode()
    cached = (CACHE / CACHED[label]).read_bytes()
    for name, raw in [(label + ".c", code), ("cached_" + label + ".c", cached)]:
        with (OUTPUT / name).open("xb") as stream:
            stream.write(raw)
    scalar = None
    if label == "rolling_forward":
        arguments = operator.arguments(vp=vp, dt=.5)
        scalar = {key: float(arguments[key]) if key == "dt" else int(arguments[key])
                  for key in ["time_m", "time_M", "x_m", "x_M", "y_m", "y_M", "dt"]}
        assert scalar == dict(time_m=1, time_M=1999, x_m=0, x_M=271, y_m=0, y_M=175, dt=.5)
        del arguments
    else:
        assert fields[0]._data is None  # No full 824MB saved-u materialization.
    records.append(dict(label=label, generated_sha256=sha(code), cached_sha256=sha(cached),
        byte_identical_to_cache=code == cached, scalar_arguments=scalar,
        saved_u_data_unallocated=label != "rolling_forward", source=label + ".c"))
report = dict(status="metadata_only_every_Operator_apply_forbidden", records=records,
    query_source_sha256=sha(Path(__file__).read_bytes()),
    environment=dict(python=platform.python_version(), numpy=np.__version__, devito=devito.__version__),
    no_Model_constructor=True, own_analytic_vp_only=True, no_BP_or_native_numeric_data_read=True,
    actual_run_cache_ancestry_claim=False,
    interpretation="Fresh pinned primary source generates these exact C bytes. Cache comparison establishes generated-source agreement, not cached-file actual-run ancestry. Rolling scalar defaults prove all logical nodes and times1..1999. Saved/gradient native per-call bounds remain separately recorded by the producer and its manufactured control.")
with (OUTPUT / "operator_metadata.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(report))
