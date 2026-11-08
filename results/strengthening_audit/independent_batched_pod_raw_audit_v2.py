"""Portable independent read-only audit of saved batched POD study artifacts.

No wavecert imports, PDE/model replay, training, new predictions or timings.
Requires numpy and the recorded Git source commit. Writes only a NEW receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METHODS = (
    "best_exact",
    "raw_POD_scalar",
    "raw_POD_batch",
    "corrected_scalar96",
    "corrected_batch96",
    "corrected_batch64",
)
CANDIDATES = ("corrected_batch96", "corrected_batch64")
PROTOCOL_SHA = "a53600eff96f128193f813c077da29235d185fca473a24e060ca8fe75334ad1f"
FILES = (
    "protocol.json",
    "summary.json",
    "status.json",
    "input_provenance.json",
    "cold_calls.csv",
    "block_quality.csv",
    "paired_cold_calls.csv",
    "cost_receipts.json",
    "parity_checks.json",
    "parity_work_receipts.json",
    "offline_reference_work.json",
    "aggregate_vectors.npz",
)
RTOL, ATOL = 1e-9, 1e-10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(first, second, message, *, rtol=RTOL, atol=ATOL):
    check(np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=False), message)


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)),
    )


def rows(path):
    with Path(path).open(newline="") as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            elif key in (
                "case_index",
                "repetition",
                "order_position",
                "block_index",
                "source_index",
            ):
                row[key] = int(value)
            elif key.endswith("_json") or key == "certificate_diagnostics":
                row[key] = json.loads(value)
            elif key not in ("parent_id", "slice_id", "method", "quality_scope"):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return result


def inside(root, relative):
    path = (root / relative).resolve()
    check(not Path(relative).is_absolute() and path.is_relative_to(root), "path outside repository")
    return path


def source_path(name):
    p = Path(name)
    return (
        name == "pyproject.toml"
        or name.startswith("src/")
        and p.suffix == ".py"
        or name.startswith("configs/")
        or name.startswith("scripts/")
        and p.suffix in (".py", ".sh")
        or p.parent.as_posix() == ".github/workflows"
        and p.suffix == ".yml"
        or p.parent.as_posix() == "."
        and p.match("requirements*.txt")
    )


def source_audit(root, summary):
    p = summary["provenance"]
    revision = p["git_revision"]
    manifest = p["source_files"]
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], cwd=root, text=True
    ).splitlines()
    check(
        set(manifest) == {name for name in tree if source_path(name)},
        "recorded source manifest incomplete for declared Git commit",
    )
    h = hashlib.sha256()
    for name, value in sorted(manifest.items()):
        historic = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        check(
            hashlib.sha256(historic).hexdigest() == value == sha(inside(root, name)),
            f"Git/current recorded-source mismatch: {name}",
        )
        h.update(name.encode() + b"\0" + value.encode() + b"\0")
    check(
        h.hexdigest() == p["source_sha256"] == summary["source_sha256_at_end"],
        "source manifest/end digest mismatch",
    )
    current = []
    current += [root / "pyproject.toml"]
    for directory in ("src", "configs", "scripts", ".github/workflows"):
        current.extend(path for path in (root / directory).rglob("*") if path.is_file())
    current.extend(root.glob("requirements*.txt"))
    added = sorted(
        path.relative_to(root).as_posix()
        for path in current
        if source_path(path.relative_to(root).as_posix())
        and path.relative_to(root).as_posix() not in manifest
    )
    return {
        "recorded_git_revision": revision,
        "source_sha256": h.hexdigest(),
        "recorded_files_verified": len(manifest),
        "all_recorded_blobs_match_git_and_current_files": True,
        "later_unrecorded_source_additions": added,
    }


def audit(root, study):
    before = {name: sha(study / name) for name in FILES}
    protocol, summary, status, inputs = [
        read_json(study / name)
        for name in ("protocol.json", "summary.json", "status.json", "input_provenance.json")
    ]
    check(
        before["protocol.json"] == PROTOCOL_SHA == summary["protocol_sha256"],
        "prospective protocol mismatch",
    )
    check(
        status["status"] == summary["status"] == "complete"
        and status["completed_cold_calls"] == 288,
        "study not complete",
    )
    check(
        summary["n_cold_calls"] == 288 and summary["n_block_quality_rows"] == 6912, "summary counts"
    )
    check(all(summary["checks"].values()), "reported study checks fail")
    check(
        protocol["methods"] == list(METHODS) and protocol["timing"]["repetitions"] == 6,
        "method/repetition contract",
    )
    check(
        not summary["calibration_or_test_arrays_loaded"]
        and not summary["external_test_arrays_loaded"]
        and not inputs["calibration_or_test_arrays_loaded"]
        and not inputs["external_test_arrays_loaded"],
        "reported role use changed",
    )
    check(
        all(
            summary["provenance"]["thread_environment"][key] == "1"
            for key in protocol["timing"]["thread_environment"]
        ),
        "thread environment contract",
    )
    sources = source_audit(root, summary)
    input_hashes = {}
    vp = protocol["validation_protocol"]
    check(
        sha(inside(root, vp["path"])) == vp["sha256"] == inputs["validation_protocol_sha256"],
        "validation protocol identity",
    )
    validation = read_json(root / vp["path"])
    for label, entry in validation["inputs"].items():
        actual = sha(inside(root, entry["path"]))
        check(
            actual == entry["sha256"] == inputs["verified_file_sha256"][label],
            f"input mismatch {label}",
        )
        input_hashes[entry["path"]] = actual
    for name, value in inputs["unchanged_production_sources"].items():
        check(sha(inside(root, name)) == value, "old source pin mismatch")
    for name, value in inputs["verified_execution_modules"].items():
        check(
            sha(inside(root, name)) == value == summary["provenance"]["source_files"][name],
            "execution module binding mismatch",
        )
    with np.load(root / validation["inputs"]["fixture"]["path"], allow_pickle=False) as f:
        fixture = {name: f[name].copy() for name in f.files}
    identities = inputs["identities"]
    check(
        len(identities) == 8 and len({r["parent_id"] for r in identities}) == 4,
        "eight slices/four parents required",
    )
    check(
        Counter(r["parent_id"] for r in identities).values() == {2}
        if False
        else all(value == 2 for value in Counter(r["parent_id"] for r in identities).values()),
        "two slices per parent",
    )
    check(
        [r["slice_id"] for r in identities] == fixture["slice_ids"].tolist()
        and [r["parent_id"] for r in identities] == fixture["parent_ids"].tolist(),
        "fixture identities",
    )
    calls, blocks, pairs = [
        rows(study / name)
        for name in ("cold_calls.csv", "block_quality.csv", "paired_cold_calls.csv")
    ]
    receipts = read_json(study / "cost_receipts.json")
    parity = read_json(study / "parity_checks.json")
    parity_cost = read_json(study / "parity_work_receipts.json")
    check(
        len(calls) == 288
        and len(blocks) == 6912
        and len(pairs) == 48
        and len(receipts) == 288
        and len(parity) == 24
        and len(parity_cost) == 48,
        "raw row counts",
    )
    expected = {(rep, case, method) for rep in range(6) for case in range(8) for method in METHODS}
    key = lambda row: (row["repetition"], row["case_index"], row["method"])  # noqa: E731 - preserve the audited AST
    bykey = {key(row): row for row in calls}
    check(len(bykey) == len(calls) and set(bykey) == expected, "complete unique288 cohort")
    blockmap = defaultdict(list)
    for row in blocks:
        blockmap[key(row)].append(row)
    check(set(blockmap) == expected, "block cohort")
    receiptmap = {key(row): row for row in receipts}
    check(len(receiptmap) == 288 and set(receiptmap) == expected, "receipt cohort")
    with np.load(study / "aggregate_vectors.npz", allow_pickle=False) as f:
        vectors = {name: f[name].copy() for name in f.files}
    check(before["aggregate_vectors.npz"] == summary["aggregate_vectors_sha256"], "vector hash")
    check(
        vectors["gradients"].shape == vectors["directions"].shape == (288, 576)
        and vectors["exact_gradients"].shape == (8, 576),
        "complete vector shapes",
    )
    check(
        vectors["gradient_row_indices"].tolist() == list(range(288)),
        "gradient mapping complete ordered",
    )
    check(
        all(
            np.all(np.isfinite(vectors[name]))
            for name in ("gradients", "directions", "exact_gradients")
        ),
        "nonfinite saved vectors",
    )
    close(
        vectors["exact_gradients"],
        fixture["exact_gradients"],
        "saved exact vectors disagree with teacher fixture",
    )
    maximum = {
        "cosine_error": 0.0,
        "gradient_norm_error": 0.0,
        "signed_derivative_error": 0.0,
        "block_signed_sum_error": 0.0,
        "timing_sum_error": 0.0,
        "bound_sum_error": 0.0,
    }
    signatures = {}
    scalar_batch_max = {"raw": 0.0, "rank96": 0.0}
    first = {}
    per_method = defaultdict(list)
    for index, row in enumerate(calls):
        rep, case, method = key(row)
        identity = identities[case]
        check(
            row["parent_id"] == identity["parent_id"] and row["slice_id"] == identity["slice_id"],
            "call identity",
        )
        shift = (rep + case) % 6
        order = METHODS[shift:] + METHODS[:shift]
        check(row["order_position"] == order.index(method), "balanced method rotation")
        check(
            vectors["method"][index] == method
            and vectors["case_index"][index] == case
            and vectors["repetition"][index] == rep,
            "vector metadata mapping",
        )
        g, d, exact = (
            vectors["gradients"][index],
            vectors["directions"][index],
            vectors["exact_gradients"][case],
        )
        norm = float(np.linalg.norm(g))
        cos = float(np.dot(g / norm, exact / np.linalg.norm(exact)))
        close(d, -g / norm, "shared normalized negative aggregate direction")
        close(row["direction_norm"], np.linalg.norm(d), "direction norm")
        for saved, derived, label in (
            (row["gradient_norm"], norm, "gradient_norm_error"),
            (row["gradient_cosine"], cos, "cosine_error"),
            (row["exact_directional_derivative"], float(exact @ d), "signed_derivative_error"),
            (row["approximate_directional_derivative"], float(g @ d), "signed_derivative_error"),
        ):
            close(saved, derived, f"vector metric {label}")
            maximum[label] = max(maximum[label], abs(saved - derived))
        close(
            row["actual_directional_error"],
            abs(float((exact - g) @ d)),
            "aggregate directional error",
        )
        check(
            row["finite_proposal"] is True
            and row["true_descent"] == bool(exact @ d < 0)
            and not row["numeric_certificate_failure"],
            "finite/true-descent flags",
        )
        times = [row[name] for name in ("setup_seconds", "query_seconds", "total_cold_seconds")]
        check(
            all(math.isfinite(value) and value >= 0 for value in times), "finite nonnegative times"
        )
        close(times[0] + times[1], times[2], "time decomposition", rtol=0, atol=1e-9)
        maximum["timing_sum_error"] = max(
            maximum["timing_sum_error"], abs(times[0] + times[1] - times[2])
        )
        check(row["valid_timing_decomposition"] is True, "timing flag")
        chosen = blockmap[key(row)]
        selected = np.flatnonzero(fixture["block_case_indices"] == case)
        check(
            len(chosen) == 24 and {r["block_index"] for r in chosen} == set(range(24)),
            "all24 block indices",
        )
        chosen = sorted(chosen, key=lambda r: r["block_index"])
        for j, block in enumerate(chosen):
            check(
                block["parent_id"] == identity["parent_id"]
                and block["slice_id"] == identity["slice_id"]
                and block["order_position"] == row["order_position"],
                "block identity",
            )
            check(
                block["source_index"] == int(fixture["source_indices"][selected[j]])
                and block["frequency_hz"] == float(fixture["frequencies_hz"][selected[j]]),
                "block source/frequency order",
            )
            check(-1 - ATOL <= block["gradient_cosine"] <= 1 + ATOL, "block cosine range")
            close(
                block["actual_directional_error"],
                abs(
                    block["exact_directional_derivative"]
                    - block["approximate_directional_derivative"]
                ),
                "block error arithmetic",
            )
        for field, target in (
            ("exact_directional_derivative", row["exact_directional_derivative"]),
            ("approximate_directional_derivative", row["approximate_directional_derivative"]),
        ):
            total = sum(block[field] for block in chosen)
            close(total, target, "sum of24 signed block derivatives")
            maximum["block_signed_sum_error"] = max(
                maximum["block_signed_sum_error"], abs(total - target)
            )
        receipt = receiptmap[key(row)]
        check(
            receipt["parent_id"] == identity["parent_id"]
            and receipt["slice_id"] == identity["slice_id"],
            "cost identity",
        )
        check(receipt["cost_units"] == summary["cost_units"], "cost unit map")
        if method == "best_exact":
            counts = receipt["query_costs"]["reference_solver"]
            rhs = sum(value for name, value in counts.items() if name.endswith("_rhs"))
            check(
                rhs == receipt["exact_expected_rhs"] == 48 and counts["factorizations"] == 4,
                "exact48 RHS/4 factors",
            )
            close(g, exact, "timed exact gradient parity")
            close(
                row["objective"], fixture["exact_objectives"][case], "timed exact objective parity"
            )
            check(row["exact_reference_parity"] is True, "exact parity flag")
        else:
            check(
                all(value == 0 for value in receipt["query_costs"]["reference_solver"].values()),
                "unexpected online fine solver work",
            )
            check(row["approximation_fine_solver_work_zero"] is True, "fine-zero flag")
            pod = receipt["query_costs"]["pod"]
            check(
                pod["reduced_forward_rhs"] == pod["reduced_adjoint_rhs"] == 24
                and pod["reduced_factorizations"] == 4,
                "POD actual RHS columns/factors",
            )
        if method.startswith("corrected_"):
            rank = 64 if method.endswith("64") else 96
            context = receipt["stability_contexts"]
            check(
                set(map(float, context)) == set(protocol["cohort"]["frequencies_hz"]),
                "four stability contexts",
            )
            mh = hashlib.sha256(fixture["models"][case].astype("<f8").ravel().tobytes()).hexdigest()
            check(receipt["model_sha256"] == mh, "context evaluated-model hash")
            available = all(value["available"] for value in context.values())
            check(row["certificate_available"] == available, "aggregate context availability")
            for frequency, ctx in context.items():
                check(
                    ctx["model_sha256"] == mh
                    and ctx["rank"] == rank
                    and ctx["n"] == 576
                    and ctx["frequency_hz"] == float(frequency),
                    "context model/rank/frequency",
                )
                sig = ctx["operator_signature"]
                check(
                    sig["rank"] == rank
                    and sig["n"] == 576
                    and sig["formally_verified_roundoff_enclosure"] is False,
                    "fixed operator/roundoff scope",
                )
                if rank in signatures:
                    check(sig == signatures[rank], "operator signature changes across cases")
                signatures[rank] = sig
                gamma = sig["first_omitted_laplacian_eigenvalue"] - (
                    2 * np.pi * float(frequency)
                ) ** 2 * float(np.max(fixture["models"][case].astype(float)))
                close(ctx["gamma"], gamma, "complement gamma algebra")
                if ctx["available"]:
                    close(ctx["tau"], ctx["k"] * ctx["ell"] / ctx["gamma"], "small-gain algebra")
                    matrix = np.array(
                        [
                            [ctx["a"], ctx["k"] / ctx["gamma"]],
                            [ctx["ell"] * ctx["a"] / ctx["gamma"], 1 / ctx["gamma"]],
                        ]
                    ) / (1 - ctx["tau"])
                    close(ctx["M"], matrix, "component matrix algebra")
                    close(
                        ctx["beta_lower_bound"],
                        1 / np.linalg.svd(matrix, compute_uv=False)[0],
                        "2x2 inverse norm algebra",
                    )
            for block in chosen:
                diagnostic = block["certificate_diagnostics"]
                bound = block["directional_error_bound"]
                check(
                    block["certificate_available"] is True
                    and bound is not None
                    and math.isfinite(bound)
                    and bound >= 0,
                    "finite available block bound",
                )
                close(
                    diagnostic["approximate_directional_derivative"],
                    block["approximate_directional_derivative"],
                    "block diagnostic center",
                )
                close(sum(diagnostic["bound_terms"]), bound, "three-term bound sum")
                cross = (
                    (2 * np.pi * block["frequency_hz"]) ** 2
                    * np.max(np.abs(d))
                    * np.linalg.norm(diagnostic["eta_u_components"])
                    * np.linalg.norm(diagnostic["eta_lambda_components"])
                )
                close(diagnostic["bound_terms"][2], cross, "independent cross term algebra")
                cover = block["actual_directional_error"] <= bound + ATOL + RTOL * max(
                    block["actual_directional_error"], bound
                )
                check(
                    block["directional_bound_covers"] == cover and cover,
                    "actual block bound misses error",
                )
            total = sum(block["directional_error_bound"] for block in chosen)
            close(total, row["directional_error_bound"], "aggregate bound sum")
            maximum["bound_sum_error"] = max(
                maximum["bound_sum_error"], abs(total - row["directional_error_bound"])
            )
            close(
                row["certified_upper_derivative"],
                row["approximate_directional_derivative"] + row["directional_error_bound"],
                "certified upper arithmetic",
            )
            cert = bool(available and row["certified_upper_derivative"] < 0)
            cover = row["actual_directional_error"] <= row[
                "directional_error_bound"
            ] + ATOL + RTOL * max(row["actual_directional_error"], row["directional_error_bound"])
            check(
                row["certified_descent"] == cert
                and row["exact_fallback_required"] == (not cert)
                and row["directional_bound_covers"] == cover
                and cover
                and row["all_block_bounds_cover"] is True,
                "aggregate certificate/coverage/fallback flags",
            )
            check(not cert or row["exact_directional_derivative"] < 0, "false certification")
            cc = receipt["query_costs"]["coarse_stability"]
            check(
                all(
                    cc[name] == 0
                    for name in ("fine_grid_rhs", "fine_grid_factorizations", "fine_grid_dense_svd")
                ),
                "coarse secretly used fine solver",
            )
            check(
                cc["coarse_inverse_rhs"] == 4 * rank
                and cc["coarse_correction_rhs"] == 48
                and cc["coarse_oriented_rhs"] == 48
                and cc["coarse_receiver_rhs"] == 80,
                "coarse logical RHS category accounting",
            )
            check(cc["coarse_solve_rhs"] == 4 * rank + 176, "coarse actual RHS total")
            for name, value in receipt["cumulative_costs"]["coarse_stability"].items():
                check(
                    value == cc[name] + receipt["setup_costs"]["coarse_stability"][name],
                    "setup+query counter decomposition",
                )
            expected_projection = 192 if method == "corrected_scalar96" else 144
            check(
                cc["coarse_projection_calls"] == cc["coarse_lift_calls"] == expected_projection
                and cc["dst_I_real_transforms"]
                == cc["idst_I_real_transforms"]
                == 2 * expected_projection,
                "projection/lift/real transform units",
            )
        else:
            check(
                all(
                    row[name] is None
                    for name in (
                        "certificate_available",
                        "certified_descent",
                        "exact_fallback_required",
                        "directional_error_bound",
                        "certified_upper_derivative",
                        "directional_bound_covers",
                        "all_block_bounds_cover",
                    )
                ),
                "uncertified control acquired a certificate",
            )
        if (case, method) in first:
            close(g, first[case, method][0], "repeat vector parity")
            close(row["objective"], first[case, method][1], "repeat objective parity")
        else:
            first[case, method] = (g, row["objective"])
        check(row["repeat_parity"] is True, "repeat flag")
        per_method[method].append(row)
    for case in range(8):
        for a, b, label in (
            ("raw_POD_scalar", "raw_POD_batch", "raw"),
            ("corrected_scalar96", "corrected_batch96", "rank96"),
        ):
            left, right = first[case, a], first[case, b]
            close(left[0], right[0], "saved timed scalar/batch gradient parity")
            close(left[1], right[1], "saved timed scalar/batch objective parity")
            scalar_batch_max[label] = max(
                scalar_batch_max[label], float(np.max(np.abs(left[0] - right[0])))
            )
    for pair in pairs:
        rep, case = pair["repetition"], pair["case_index"]
        for method in METHODS:
            for field in ("setup_seconds", "query_seconds", "total_cold_seconds"):
                close(
                    pair[f"{method}_{field}"],
                    bykey[rep, case, method][field],
                    "paired raw timing map",
                )
    expected_parity = {
        (case, label)
        for case in range(8)
        for label in ("raw_POD_batch", "corrected_batch96", "corrected_batch64")
    }
    check(
        {(row["case_index"], row["batch_quality"]["method"]) for row in parity} == expected_parity,
        "complete preflight comparisons",
    )
    for row in parity:
        check(
            row["passes"]
            and all(
                side["passes"] and all(side["fields"].values())
                for side in row["finite_fields"].values()
            ),
            "preflight finite-field gate",
        )
        check(
            all(value["passes"] for value in row["comparisons"].values()),
            "preflight reported comparison failure",
        )
        for value in row["comparisons"].values():
            if "maximum_absolute_error" in value and value["maximum_absolute_error"] is not None:
                check(
                    math.isfinite(value["maximum_absolute_error"])
                    and math.isfinite(value["relative_l2_error"])
                    and value["maximum_absolute_error"] <= ATOL,
                    "saved preflight max-absolute error exceeds sufficient atol bound",
                )
        timed = bykey[0, row["case_index"], row["batch_quality"]["method"]]
        for field in (
            "objective",
            "gradient_norm",
            "gradient_cosine",
            "exact_directional_derivative",
            "approximate_directional_derivative",
        ):
            close(row["batch_quality"][field], timed[field], "preflight versus saved timed quality")
        check(
            row["batch_quality"]["certified_descent"] == timed["certified_descent"],
            "preflight/timed strict certificate",
        )
    timing = {}
    eligibility = {}
    slices = {}
    for method, group in per_method.items():
        parents = {}
        for parent in sorted({row["parent_id"] for row in group}):
            case_indices = sorted(
                {row["case_index"] for row in group if row["parent_id"] == parent}
            )
            parents[parent] = float(
                np.mean(
                    [
                        np.median(
                            [bykey[rep, case, method]["total_cold_seconds"] for rep in range(6)]
                        )
                        for case in case_indices
                    ]
                )
            )
        score = float(np.mean(list(parents.values())))
        timing[method] = {
            "equal_parent_mean_slice_median_total_cold_ms": 1000 * score,
            "parent_means_ms": {k: 1000 * v for k, v in parents.items()},
            "global_median_ms": 1000
            * float(np.median([row["total_cold_seconds"] for row in group])),
        }
        close(
            score,
            summary["descriptive_methods"][method][
                "equal_parent_mean_slice_median_total_cold_seconds"
            ],
            "equal-parent descriptive timing score",
        )
        if method.startswith("corrected_"):
            first_rows = [bykey[0, case, method] for case in range(8)]
            slices[method] = {
                "certified_unique_slices": sum(row["certified_descent"] for row in first_rows),
                "available_unique_slices": sum(row["certificate_available"] for row in first_rows),
                "true_descent_unique_slices": sum(row["true_descent"] for row in first_rows),
                "false_certifications": sum(
                    row["certified_descent"] and not row["true_descent"] for row in group
                ),
                "uncertified_slice_ids": [
                    row["slice_id"] for row in first_rows if not row["certified_descent"]
                ],
            }
        if method in CANDIDATES:
            eligible = all(
                row["finite_proposal"]
                and row["repeat_parity"]
                and row["certificate_available"]
                and row["certified_descent"]
                and row["directional_bound_covers"]
                and row["all_block_bounds_cover"]
                for row in group
            )
            eligibility[method] = {
                "eligible": bool(eligible),
                "score_seconds": score,
                "parent_means_seconds": parents,
            }
            saved = summary["development_recommendation"]["candidates"][method]
            check(
                saved["complete"] is True and saved["eligible"] == eligible,
                "candidate complete/eligibility",
            )
            close(
                saved["equal_parent_mean_slice_median_total_cold_seconds"],
                score,
                "candidate cost score",
            )
            for parent, value in parents.items():
                close(
                    saved["parent_means_of_slice_medians"][parent], value, "candidate parent scores"
                )
    eligible = [method for method in CANDIDATES if eligibility[method]["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda method: (eligibility[method]["score_seconds"], CANDIDATES.index(method)),
        )
        if eligible
        else None
    )
    check(
        summary["development_recommendation"]["recommended_development_candidate"] == winner
        and summary["development_recommendation"]["automatic_deployment_or_heldout_evaluation"]
        is False,
        "development-only recommendation",
    )
    check(
        all(sha(study / name) == value for name, value in before.items()),
        "raw artifact changed during audit",
    )
    check(
        all(sha(inside(root, name)) == value for name, value in input_hashes.items()),
        "input changed during audit",
    )
    return {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha(Path(__file__)),
        "protocol_sha256": PROTOCOL_SHA,
        "source": sources,
        "raw_file_sha256": before,
        "input_file_sha256": input_hashes,
        "verified_counts": {
            "cold_calls": 288,
            "block_rows": 6912,
            "paired_rows": 48,
            "preflight_comparisons": 24,
            "cost_receipts": 288,
            "finite_vector_rows": 288,
            "unique_slices": 8,
            "parent_groups": 4,
        },
        "maximum_arithmetic_discrepancies": maximum,
        "maximum_saved_scalar_batch_gradient_absolute_difference": scalar_batch_max,
        "timing": timing,
        "corrected_slice_counts": slices,
        "candidate_eligibility": eligibility,
        "recomputed_development_recommendation": winner,
        "ratios": {
            "batch96_over_scalar96": timing["corrected_batch96"][
                "equal_parent_mean_slice_median_total_cold_ms"
            ]
            / timing["corrected_scalar96"]["equal_parent_mean_slice_median_total_cold_ms"],
            "batch96_over_exact": timing["corrected_batch96"][
                "equal_parent_mean_slice_median_total_cold_ms"
            ]
            / timing["best_exact"]["equal_parent_mean_slice_median_total_cold_ms"],
            "raw_batch_over_exact": timing["raw_POD_batch"][
                "equal_parent_mean_slice_median_total_cold_ms"
            ]
            / timing["best_exact"]["equal_parent_mean_slice_median_total_cold_ms"],
        },
        "PDE_or_model_replays": 0,
        "main_or_raw_artifact_mutations": 0,
        "limitations": [
            "Independent stored-artifact arithmetic/cohort/source/accounting audit, not a rerun of PDE/model values.",
            "Saved aggregate vectors permit independent cosine/direction/derivative sums and repeat/scalar-batch checks. Full per-block gradient/state/adjoint fields are not saved; their preflight parity relies on recorded residual/error summaries and existing tests.",
            "Raw approximate gradient, exact full gradient and certified physical proposal are different information tasks. No inversion/equal-accuracy/time-to-target or heldout promotion follows.",
            "Eight reused development slices/four parent groups with six timing visits; descriptive timings, not IID geological inference or proof of OS idleness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=Path("results/batched_pod_development"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    study = args.study.resolve() if args.study.is_absolute() else (root / args.study).resolve()
    check(not args.output.exists(), "new audit receipt path required")
    try:
        receipt = audit(root, study)
    except Exception as error:
        receipt = {
            "status": "failed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "audit_script_sha256": sha(Path(__file__)),
            "error_type": type(error).__name__,
            "error": str(error),
            "PDE_or_model_replays": 0,
            "main_or_raw_artifact_mutations": 0,
        }
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.output),
                "counts": receipt["verified_counts"],
                "timing_ms": {
                    k: v["equal_parent_mean_slice_median_total_cold_ms"]
                    for k, v in receipt["timing"].items()
                },
                "corrected_slice_counts": receipt["corrected_slice_counts"],
                "recommendation": receipt["recomputed_development_recommendation"],
                "ratios": receipt["ratios"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
