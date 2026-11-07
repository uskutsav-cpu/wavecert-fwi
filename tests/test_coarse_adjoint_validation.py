import copy
import json
from pathlib import Path

import numpy as np
import pytest

from wavecert.experiments.coarse_adjoint_validation import (
    _cosine,
    _ratio,
    load_coarse_adjoint_inputs,
    run_coarse_adjoint_validation,
    validate_coarse_adjoint_protocol,
)
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.pod import PODGalerkinSurrogate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/coarse_adjoint_validation_protocol.json"


def test_compact_fixture_and_all_frozen_provenance_verify_without_any_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input checks must not construct a surrogate or solve physics")

    monkeypatch.setattr(PODGalerkinSurrogate, "__init__", forbidden)
    monkeypatch.setattr(Helmholtz2D, "_solve", forbidden)
    arrays, basis, audit = load_coarse_adjoint_inputs(json.loads(PROTOCOL.read_bytes()), ROOT)
    assert basis.shape == (576, 64)
    assert len(set(arrays["parent_ids"])) == 4
    assert len(set(arrays["slice_ids"])) == 8
    assert audit["calibration_or_test_arrays_loaded"] is False
    assert audit["external_test_arrays_loaded"] is False
    assert audit["shared_offline_teacher_label_generation"]["validation_inverse_cases"]["n_blocks"] == 192


@pytest.mark.parametrize("key,value", [
    ("spectral_coarse_rank", 64), ("methods", ["coarse_corrected_oriented"]),
    ("source_indices", [50]), ("heldout_arrays_loaded", True),
])
def test_fixed_protocol_rejects_rank_method_cohort_and_role_changes(key, value):
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol[key] = value
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_coarse_adjoint_protocol(protocol)


def test_pinned_input_hash_mismatch_fails_before_archive_use():
    protocol = copy.deepcopy(json.loads(PROTOCOL.read_bytes()))
    protocol["inputs"]["fixture"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA-256 mismatch: fixture"):
        load_coarse_adjoint_inputs(protocol, ROOT)


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    previous = output / "summary.json"
    previous.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="new or empty"):
        run_coarse_adjoint_validation(PROTOCOL, output, root=ROOT)
    assert previous.read_bytes() == b"preserve me"


def test_missing_and_large_finite_proposal_diagnostics_remain_json_safe():
    exact = np.array([1e308, -1e308])
    assert _cosine(exact, None) is None and _cosine(None, exact) is None
    assert _cosine(exact, exact) == pytest.approx(1)
    assert _cosine(exact, np.zeros(2)) is None
    assert _ratio(1e308, 1e-308) is None
    assert _ratio(1., 0.) is None
    json.dumps({"ratio": _ratio(1e308, 1e-308), "cosine": _cosine(exact, exact)}, allow_nan=False)
