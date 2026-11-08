"""One-off old-data binding/Decimal review; no producer import or solver calls."""
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "work/wavecert-fwi"
OUTPUT = ROOT / "work/new-phase-staging/stability_anchor_feasibility_independent_review.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def model_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def atan_inverse(n):
    x, result, sign, k = Decimal(1) / n, Decimal(0), 1, 0
    power = x
    while True:
        term = power / (2 * k + 1)
        result += sign * term
        if abs(term) < Decimal("1e-92"):
            return result
        power *= x * x
        sign, k = -sign, k + 1


def main():
    stage = ROOT / "work/new-phase-staging/stability_anchor_feasibility"
    source, registration = stage / "assess_anchor.py", stage / "registration.json"
    result_path = ROOT / "work/stability_anchor_feasibility_01/report.json"
    freeze = load(ROOT / "work/stability_anchor_feasibility_prefreeze.json")
    rules, result = load(registration), load(result_path)
    assert sha(source) == freeze["source_sha256"] == result["source_sha256"]
    assert sha(registration) == freeze["registration_sha256"] == result["registration_sha256"]
    pins = {path: sha(REPO / path) for path in result["input_sha256"]}
    assert pins == result["input_sha256"]
    witness, cohort = REPO / rules["witness_directory"], REPO / rules["cohort_directory"]
    wr, cr = load(witness / "complete_receipt.json"), load(cohort / "complete_receipt.json")
    assert sha(witness / "complete_receipt.json") == rules["witness_complete_sha256"]
    assert sha(cohort / "complete_receipt.json") == rules["cohort_complete_sha256"]
    assert wr["source_before"] == wr["source_after"] and cr["source_before"] == cr["source_after"]
    source9f = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
    assert wr["source_before"]["sha256"] == cr["source_before"]["production"]["repository_source_sha256"] == source9f
    operator_path = REPO / "src/wavecert/physics/helmholtz.py"
    operator_sha = sha(operator_path)
    endpoint_rows = {row["run_index"]: row for row in cr["runs"]}
    reported_rows = {row["run_index"]: row for row in result["rows"]}
    assert len(endpoint_rows) == len(reported_rows) == 48 and set(endpoint_rows) == set(reported_rows)
    assert {(r["case_index"], r["method"], r["repetition"]) for r in cr["runs"]} == {
        (c, m, r) for c in range(8) for m in ["native_m_lbfgsb", "physical_v_lbfgsb"] for r in range(3)}
    anchors = {(r["case_index"], r["frequency_hz"]): r for r in result["anchors"]}
    assert len(anchors) == 32
    margins, max_distance_error, max_penalty_error, nonidentical_common_starts = [], Decimal(0), Decimal(0), 0
    with localcontext() as ctx:
        ctx.prec = 95
        pi = 16 * atan_inverse(Decimal(5)) - 4 * atan_inverse(Decimal(239))
        for case in range(8):
            record = load(witness / f"case_{case:02d}_receipt.json")
            with np.load(witness / record["raw_arrays_artifact"], allow_pickle=False) as bundle:
                anchor_model = bundle["initial_model"].copy()
            binding = record["production"]["certificate"]["binding"]
            anchor_sha = model_sha(anchor_model)
            assert anchor_sha == binding["model"]["sha256"] == binding["model_sha256"]
            assert binding["physics_config"] == rules["physics"]
            frequencies = record["production"]["certificate"]["frequency_records"]
            assert [f["frequency_hz"] for f in frequencies] == rules["frequencies_hz"]
            for endpoint in [r for r in cr["runs"] if r["case_index"] == case]:
                index, provenance = endpoint["run_index"], endpoint["output_provenance"]
                contract = provenance["policy_contract"]
                assert endpoint["status"] == "complete"
                assert contract["data_sha256"] == binding["data_sha256"] and contract["geometry_sha256"] == binding["geometry_sha256"]
                assert contract["reference_source_sha256"] == contract["operator_base_record"]["source_sha256"] == operator_sha
                assert contract["operator_base_record"]["configuration"] == rules["physics"]
                assert contract["operator_base_record"]["damping"] == binding["damping"]
                assert provenance["common_start_record"]["requested_native_model_sha256"] == anchor_sha
                nonidentical_common_starts += not provenance["common_start_record"]["native_roundtrip_bitwise_equal"]
                with np.load(cohort / endpoint["models_artifact"], allow_pickle=False) as bundle:
                    final = bundle["final_native_m"].copy()
                assert model_sha(final) == provenance["final_actual_native_m_sha256"]
                assert final.shape == anchor_model.shape and np.isfinite(final).all() and (final > 0).all()
                distance = max(abs(Decimal.from_float(float(a)) - Decimal.from_float(float(b)))
                               for a, b in zip(final.ravel(), anchor_model.ravel(), strict=True))
                reported = reported_rows[index]
                assert all(reported[k] == endpoint[k] for k in ["case_index", "method", "repetition"])
                error = abs(Decimal.from_float(reported["model_displacement_infinity"]) - distance)
                max_distance_error = max(max_distance_error, error)
                assert error <= Decimal("1e-14") * max(Decimal(1), distance)
                assert [r["frequency_hz"] for r in reported["frequency_bounds"]] == rules["frequencies_hz"]
                for frequency, candidate in zip(frequencies, reported["frequency_bounds"], strict=True):
                    f, stability = frequency["frequency_hz"], frequency["stability"]
                    assert stability["available"] and stability["model_sha256"] == anchor_sha and stability["physics_config"] == rules["physics"]
                    assert stability["frequency_hz"] == f and stability["operator_signature"]["damping_float64_le_sha256"] == binding["damping"]["sha256"]
                    assert stability["numerical_diagnostics"]["formally_verified_roundoff_enclosure"] is False
                    beta = Decimal.from_float(stability["beta_lower_bound"])
                    assert beta > 0 and candidate["beta_anchor"] == anchors[case, f]["beta_anchor"] == float(beta)
                    omega2 = (2 * pi * Decimal.from_float(f)) ** 2
                    penalty, margin = omega2 * distance, beta - omega2 * distance
                    rel_error = abs(Decimal.from_float(candidate["perturbation_penalty"]) - penalty) / penalty
                    max_penalty_error = max(max_penalty_error, rel_error)
                    assert rel_error < Decimal("1e-14")
                    radius = beta / omega2
                    assert abs(Decimal.from_float(anchors[case, f]["positive_reuse_radius"]) - radius) < Decimal("1e-14") * radius
                    assert margin < 0 and candidate["candidate_lower_bound"] == 0
                    margins.append(dict(run_index=index, case_index=case, frequency_hz=f, Decimal_margin=str(margin)))
        assert len(margins) == result["frequency_endpoint_denominator"] == 192
        assert result["endpoint_denominator"] == 48 and result["positive_instances"] == 0 and result["zero_instances"] == 192
        assert result["unavailable_anchor_instances"] == 0
        review = dict(status="CLEAR", role="AI independent old-data conditional-inequality and arithmetic review",
            arithmetic="95-digit Decimal Machin-pi and exact decoded-binary64 model differences; no producer import or solver call; ordinary high-precision check, not directed interval arithmetic",
            anchors_available=32, endpoint_models_checked=48, frequency_endpoint_instances_checked=192,
            positive_conditional_candidates=0, zero_conditional_candidates=192,
            largest_signed_margin=max((r["Decimal_margin"] for r in margins), key=Decimal),
            max_distance_absolute_disagreement=str(max_distance_error), max_penalty_relative_disagreement=str(max_penalty_error),
            nonidentical_requested_vs_common_FP_starts=nonidentical_common_starts,
            exact_arithmetic_argument="For every complex unitx, ||A_final x||2>=||A_anchor x||2-||E x||2>=beta_anchor-||E||2. Here E=-omega²diag(m_final-m_anchor), so ||E||2=omega²||delta_m||infinity. sigma_min is nonnegative. This requires fixed grid/stiffness/damping and a genuinely valid anchor lower bound; normality/Hermitian structure is not required.",
            limits=["All anchors are ordinary-FP estimates with unverified PSD allowances, not formal lower enclosures.",
                "Zero means this sufficient lower bound is unavailable; it does not establish actual instability.",
                "Stored anchor is the requested initial native-m witness; actual common FP starts can differ. Distance here explicitly uses the bound's stored anchor, not a borrowed receipt for the common start.",
                "Final snapshots establish neither intermediate reuse, refresh policy, online cost nor new controller performance.",
                "This is a source-informed standard inequality, not novelty, independent confirmation or a student final research design. No BP arrays, PDEs or optimizers were evaluated."],
            source_pins=dict(producer=sha(source), registration=sha(registration), prefreeze=sha(ROOT / "work/stability_anchor_feasibility_prefreeze.json"),
                assessed_report=sha(result_path), operator_source=operator_sha, reviewer_source=sha(Path(__file__))),
            old_input_files_sha256=pins, all192_margin_receipts=margins)
    assert pins == {path: sha(REPO / path) for path in pins}
    with OUTPUT.open("x") as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: review[k] for k in ["status", "anchors_available", "endpoint_models_checked", "frequency_endpoint_instances_checked", "largest_signed_margin"]}))


if __name__ == "__main__":
    main()
