from pathlib import Path

import numpy as np

from wavecert.benchmarks.marmousi import load_marmousi_binary
from wavecert.benchmarks.openfwi import load_openfwi_velocity_batch
from wavecert.benchmarks.procedural import GEOLOGY_SETTINGS, procedural_geology


def test_marmousi_binary_loader(tmp_path: Path):
    arr = np.linspace(1500.0, 3000.0, 12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "marm.bin"
    arr.tofile(path)
    loaded = load_marmousi_binary(path, shape=(3, 4))
    assert loaded.shape == (3, 4)
    assert np.allclose(loaded, arr)


def test_openfwi_velocity_loader(tmp_path: Path):
    arr = np.ones((2, 1, 7, 7), dtype=np.float32)
    path = tmp_path / "model1.npy"
    np.save(path, arr)
    loaded = load_openfwi_velocity_batch(path)
    assert loaded.shape == (2, 7, 7)


def test_all_procedural_geology_settings_are_finite():
    rng = np.random.default_rng(1)
    for setting in GEOLOGY_SETTINGS:
        model = procedural_geology(setting, (16, 16), rng)
        assert model.shape == (16, 16)
        assert np.isfinite(model).all()
        assert np.all(model > 0)
