"""Metadata only: every Operator.apply is forbidden; never propagate waves."""
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from devito import Function, Grid, TimeFunction
from devito.operator.operator import Operator as BaseOperator
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic.operators import ForwardOperator
from examples.seismic.model import PhysicalDomain

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def forbidden(*args, **kwargs):
    raise RuntimeError("Every Operator.apply is forbidden in metadata-only construction")

BaseOperator.apply = forbidden
records = []
for damped in [False, True]:
    grid = Grid(shape=(61, 61), extent=(600., 600.), dtype=np.float64,
                subdomains=(PhysicalDomain(4, fs=False),))
    velocity = Function(name="vp", grid=grid, space_order=4)
    velocity.data[:] = 2.  # Own analytic initialization, no primary field read.
    damping = Function(name="damp", grid=grid, space_order=4) if damped else 0
    model = SimpleNamespace(dim=2, grid=grid, critical_dt=2.773,
        m=1/(velocity*velocity), damp=damping, fs=False, spacing_map=grid.spacing_map)
    geometry = AcquisitionGeometry(model, np.array([[350., 300.]]),
        np.array([[300., 300.]]), 0., 1100., f0=.005, src_type="Ricker")
    geometry.resample(1.)
    operator = ForwardOperator(model, geometry, space_order=4, save=False)
    arguments = operator.arguments(vp=velocity, dt=1.)
    # Read scalar metadata ONLY, never source/receiver/wavefield argument data.
    scalar = {key: float(arguments[key]) if key == "dt" else int(arguments[key])
        for key in ["time_m", "time_M", "x_m", "x_M", "y_m", "y_M", "dt"]}
    assert scalar == dict(time_m=1, time_M=1099, x_m=0, x_M=60, y_m=0, y_M=60, dt=1.)
    code = str(operator.ccode).encode()
    name = "damped" if damped else "undamped"
    code_path = ROOT/("metadata_query_"+name+"_forward.c")
    with code_path.open("xb") as stream:
        stream.write(code)
    primary = (ROOT/("forward_"+name+"_primary.c")).read_bytes()
    records.append(dict(mode=name, scalar_arguments=scalar,
        generated_code_sha256=sha(code), cached_primary_code_sha256=sha(primary),
        byte_identical_to_cached_primary=code == primary,
        stripped_text_identical_to_cached_primary=code.strip() == primary.strip()))
symbolic = TimeFunction(name="symbolic_u_no_propagation", grid=grid,
                        time_order=2, space_order=4)
result = dict(status="metadata_only_no_Operator_apply_or_wave_propagation",
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    query_source_sha256=sha(Path(__file__).read_bytes()), records=records,
    symbolic_derivatives=dict(dt=str(symbolic.dt.evaluate),
        dt2=str(symbolic.dt2.evaluate), dx2=str(symbolic.dx2.evaluate)),
    source_receiver_or_wavefield_values_read=False, cached_numerical_values_read=False,
    own_analytic_velocity_initialization=True, Model_constructor_called=False,
    interpretation="Pinned primary operator argument defaults, reference-informed verification port. Loop t=1..1099 inclusive; initial u0,u1 zero; stencil and source update u(t+1), then rec[t]=u(t). Receiver indices0 and1100 untouched. This is not a blind derivation or physical validation.",
    earlier_guarded_attempt="Actual Model constructor even with nbl0 tried an Operator.apply for velocity initialization; it was blocked before any apply. Pure Grid/Function metadata construction avoids Model and initdamp entirely.")
with (ROOT/"operator_time_arguments.json").open("x") as stream:
    json.dump(result, stream, indent=2); stream.write("\n")
print(json.dumps(result))
