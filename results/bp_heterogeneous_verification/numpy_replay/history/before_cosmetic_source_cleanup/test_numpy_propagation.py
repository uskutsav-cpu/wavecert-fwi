"""Small manufactured rolling-trajectory controls, never BP inputs."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("_manufactured_bp_numpy_rolling", Path(__file__).with_name("numpy_bp_tangent.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def manufactured():
    x, z = np.indices((13, 12))
    mass = .2 + .002*x + .001*z
    directions = np.asarray([np.ones_like(mass), 1.+.2*np.cos(x+.3*z)])
    directions /= np.linalg.norm(directions, axis=(1, 2))[:, None, None]
    settings = dict(width=3, dt=.5, nt=161, source_node=(6, 5), receiver_nodes=[(6, 5), (5, 5), (7, 6)])
    return mass, directions, settings


def test_variable_model_rolling_tangents_match_two_fixed_central_differences():
    mass, directions, settings = manufactured()
    result, _ = core.propagate(1/np.sqrt(mass), directions, **settings)
    empty = np.zeros((0, *mass.shape))
    for index, direction in enumerate(directions):
        for epsilon in [1e-5, 3e-6]:
            plus, _ = core.propagate(1/np.sqrt(mass+epsilon*direction), empty, **settings)
            minus, _ = core.propagate(1/np.sqrt(mass-epsilon*direction), empty, **settings)
            fd = (plus["receiver"]-minus["receiver"])/(2*epsilon)
            expected = result["tangent_receiver"][index]
            assert np.linalg.norm(expected) > 0
            assert np.linalg.norm(fd-expected)/np.linalg.norm(expected) < 2e-5


def test_recording_current_slot_and_untouched_end_sentinels():
    mass, directions, settings = manufactured()
    result, record = core.propagate(1/np.sqrt(mass), directions, **settings)
    assert record["time_bounds"] == dict(time_m=1, time_M=159)
    assert np.all(result["receiver"][[0, 1, -1]] == 0)
    assert np.all(result["tangent_receiver"][:, [0, 1, -1]] == 0)
    assert result["receiver"][2, 0] != 0
    assert result["tangent_receiver"].shape == (2, 161, 3)
    assert record["full_wavefield_history_saved"] is False
    assert np.array_equal(result["computed_mask"], (np.arange(161)>=1)&(np.arange(161)<=159))
