"""AI post-outcome initial-linearization diagnosis; default reads metadata only.

This independently assembles the old49x49 frequency-domain graph. No production
or witness module is imported. Project solves require a separate root GO and
exact source/protocol hashes; the code supplies no optimization or adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def safe(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping registered input: " + str(name))
    return path


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_digest(root):
    files = [root / "pyproject.toml"] + list((root / "src").rglob("*.py"))
    files += [p for p in (root / "configs").rglob("*") if p.is_file()]
    files += [p for p in (root / "scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    files += list((root / ".github/workflows").glob("*.yml")) + list(root.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + sha(path).encode() + b"\0")
    return digest.hexdigest()


def descriptor(array):
    import numpy as np

    canonical = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return dict(shape=list(array.shape), dtype=canonical.dtype.str,
                sha256=hashlib.sha256(canonical.tobytes()).hexdigest(), raw_values_saved=True)


def norm(array):
    return math.hypot(*(part for value in array.ravel()
                        for part in (float(value.real), float(value.imag))))


def real_dot(a, b):
    return math.fsum(float(x.real)*float(y.real) + float(x.imag)*float(y.imag)
                     for x, y in zip(a.ravel(), b.ravel(), strict=True))


def diagnostics(truth_receiver, initial_receiver, tangent_receiver):
    """No selection or physical threshold; zero denominators remain unknown."""
    correction = truth_receiver-initial_receiver
    remainder = correction-tangent_receiver
    data_norm, tangent_norm, remainder_norm = norm(correction), norm(tangent_receiver), norm(remainder)
    alignment_scale = data_norm*tangent_norm
    return dict(data_change_norm=data_norm, tangent_norm=tangent_norm, remainder_norm=remainder_norm,
        remainder_relative_to_data_change=remainder_norm/data_norm if data_norm else None,
        zero_data_change=data_norm == 0, zero_tangent=tangent_norm == 0,
        real_alignment=real_dot(correction, tangent_receiver)/alignment_scale if alignment_scale else None,
        initial_objective=.5*data_norm**2, linearized_truth_step_objective=.5*remainder_norm**2,
        g_data_dot_delta=real_dot(initial_receiver-truth_receiver, tangent_receiver),
        interpretation="one fixed full truth-correction linearization; not an optimization step or causal diagnosis")


def graph(shape, spacing):
    """Direct neighbor graph including diagonal4/h^2 on boundary unknowns."""
    import numpy as np
    import scipy.sparse as sp

    nz, nx = shape
    rows, cols, values = [], [], []
    for z in range(nz):
        for x in range(nx):
            index = z*nx+x
            rows.append(index)
            cols.append(index)
            values.append(4/spacing**2)
            for zz, xx in ((z-1, x), (z+1, x), (z, x-1), (z, x+1)):
                if 0 <= zz < nz and 0 <= xx < nx:
                    rows.append(index)
                    cols.append(zz*nx+xx)
                    values.append(-1/spacing**2)
    distance = np.asarray([min(z, x, nz-z-1, nx-x-1) for z in range(nz) for x in range(nx)])
    damping = 2*np.maximum(0, (6-distance)/6)**2
    return sp.coo_matrix((values, (rows, cols)), shape=(nz*nx, nz*nx)).tocsc(), damping


def frequency_probe(initial, truth, stiffness, damping, rhs, receivers, frequency,
                    operations, on_update=lambda: None, on_receiver=lambda name, value: None):
    """Two LU factorizations,18 RHS solves at six sources, no adjoint solve."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    omega = 2*math.pi*frequency
    omega2 = omega*omega
    matrices = {name: stiffness+sp.diags(-omega2*model+1j*omega*damping, format="csc")
                for name, model in (("truth", truth), ("initial", initial))}
    factors, fields = {}, {}
    for name in ("truth", "initial", "tangent"):
        row = operations[name]
        row.update(status="attempted", rhs_columns_attempted=0, rhs_columns_completed=0)
        if name != "tangent":
            row["factorizations_attempted"] = 1
        on_update()
        try:
            if name != "tangent":
                factors[name] = spla.splu(matrices[name])
                row["factorizations_completed"] = 1
                forcing, matrix = rhs, matrices[name]
                row["rhs_columns_attempted"] = rhs.shape[1]
                on_update()
                field = factors[name].solve(forcing)
            else:
                forcing = omega2*(truth-initial)[:, None]*fields["initial"]
                matrix = matrices["initial"]
                row["rhs_columns_attempted"] = rhs.shape[1]
                on_update()
                field = factors["initial"].solve(forcing)
            row["rhs_columns_completed"] = rhs.shape[1]
            receiver = field[receivers, :].copy()
            on_receiver(name, receiver)
            if not np.isfinite(field).all():
                raise FloatingPointError("nonfinite field")
            defect = matrix@field-forcing
            residuals = [norm(defect[:, j])/max(norm(forcing[:, j]), 1e-300) for j in range(rhs.shape[1])]
            row["true_relative_residuals"] = [v if math.isfinite(v) else None for v in residuals]
            if any(not math.isfinite(v) or v > 1e-9 for v in residuals):
                raise ValueError("fixed1e-9 true residual guard failed")
            fields[name] = field
            row["status"] = "complete"
        except BaseException as exc:
            row.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                       error=f"{type(exc).__name__}: {exc}", incomplete_linear_work_unknown=True)
            raise
        finally:
            on_update()
    return diagnostics(fields["truth"][receivers], fields["initial"][receivers], fields["tangent"][receivers])


def planned(protocol):
    return [dict(case_index=i, status="unattempted", bands={name: dict(status="unavailable") for name in protocol["bands_hz"]},
        frequencies=[dict(frequency_hz=f, status="unattempted", operations={name: dict(status="unattempted",
            rhs_columns_planned=6, rhs_columns_attempted=0, rhs_columns_completed=0,
            factorizations_attempted=0, factorizations_completed=0) for name in ("truth", "initial", "tangent")})
            for f in protocol["frequencies_hz"]]) for i in range(8)]


def load_protocol(path):
    p = json.loads(path.read_bytes())
    if (p["frequencies_hz"] != [.25, .5, 1., 2., 2.5, 3., 3.5, 4.]
            or p["bands_hz"] != dict(lowband=[.25, .5, 1., 2.], originalband=[2.5, 3., 3.5, 4.])
            or p["case_indices"] != list(range(8)) or p["planned_rhs_solves"] != 1152
            or p["original_band_DD_relative_allowance"] != 1e-9):
        raise ValueError("fixed frequencies, bands, cases, denominator or DD allowance changed")
    return p


def execute(args, p):
    import numpy as np

    witness, output = args.witness.resolve(), args.output.resolve()
    if output.exists() or any(output.is_relative_to(v) for v in (witness, args.repo.resolve(), args.protocol.resolve().parent)):
        raise ValueError("fresh outer output outside original artifacts/source required")
    if sha(safe(witness, "complete_receipt.json")) != p["raw_initial_complete_receipt_sha256"]:
        raise ValueError("frozen complete witness receipt changed")
    complete = json.loads((witness/"complete_receipt.json").read_bytes())
    if complete["status"] != "complete_initial_direction_verification" or complete["verification_result"] != "clear":
        raise ValueError("complete old witness required")
    for name, digest in p["input_artifacts_sha256"].items():
        if sha(safe(witness, name)) != digest or complete["artifact_sha256"][name] != digest:
            raise ValueError("registered witness artifact changed: " + name)
    if source_digest(args.repo) != p["production_source_sha256"]:
        raise ValueError("production source9f changed")
    output.mkdir()
    status = dict(status="running", scope=p["scope"], source_sha256=sha(Path(__file__)), protocol_sha256=sha(args.protocol),
        started_utc=datetime.now(timezone.utc).isoformat(), physical_acceptance=None, cases=planned(p),
        planned_rhs_solves=1152, planned_frequencies=64, planned_cases=8, optimizer_calls=0,
        original_complete_sha256=p["raw_initial_complete_receipt_sha256"])
    def persist():
        atomic(output/"status.json", status)
    persist()
    try:
        original = json.loads(safe(witness, "original_protocol.json").read_bytes())
        stiffness, damping = graph((49, 49), .025)
        def mapped(old):
            return (2*(old//24)+1)*49+2*(old%24)+1
        sources = np.asarray([mapped(i) for i in original["geometry"]["original_source_indices"]], dtype=np.int64)
        receivers = np.asarray([mapped(i) for i in original["geometry"]["original_receiver_indices"]], dtype=np.int64)
        rhs = np.zeros((2401, 6), dtype=np.complex128)
        rhs[sources, np.arange(6)] = 1/.025**2
        for row in status["cases"]:
            index = row["case_index"]
            row["status"] = "attempted"
            persist()
            try:
                receipt = json.loads(safe(witness, f"case_{index:02d}_receipt.json").read_bytes())
                if receipt["status"] != "complete" or not receipt["raw_witness_complete"] or receipt["case_index"] != index:
                    raise ValueError("raw witness incomplete")
                names = ["initial_model", "independent_truth_model", "independent_true_gradient", "independent_damping",
                         "source_rhs", "source_indices", "receiver_indices"]
                with np.load(safe(witness, f"case_{index:02d}_arrays.npz"), allow_pickle=False) as bundle:
                    arrays = {name: bundle[name].copy() for name in names}
                if any(descriptor(a) != receipt["arrays"][name] for name, a in arrays.items()):
                    raise ValueError("raw canonical vector descriptor mismatch")
                initial, truth, gradient = (arrays[k] for k in names[:3])
                if any(a.shape != (2401,) or a.dtype != np.float64 or not np.isfinite(a).all() for a in (initial, truth, gradient)):
                    raise ValueError("model/gradient vector schema")
                if (np.any(initial <= 0) or np.any(truth <= 0) or not np.array_equal(damping, arrays["independent_damping"])
                        or not np.array_equal(rhs, arrays["source_rhs"]) or not np.array_equal(sources, arrays["source_indices"])
                        or not np.array_equal(receivers, arrays["receiver_indices"])):
                    raise ValueError("positive model/damping/source normalization/order changed")
                row["input_vectors"] = {name: receipt["arrays"][name] for name in names}
                by_frequency = {}
                for frequency_row in row["frequencies"]:
                    frequency = frequency_row["frequency_hz"]
                    frequency_row["status"] = "attempted"
                    retained = {}
                    def retain(name, receiver, retained=retained):
                        retained[name+"_receiver"] = receiver
                    persist()
                    try:
                        result = frequency_probe(initial, truth, stiffness, damping, rhs, receivers, frequency,
                                                 frequency_row["operations"], persist, retain)
                        frequency_row.update(status="complete", diagnostics=result)
                        by_frequency[frequency] = retained
                    except Exception as exc:
                        frequency_row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    except BaseException as exc:
                        frequency_row.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        if retained:
                            filename = f"case_{index:02d}_f{frequency:g}_receiver.npz"
                            with (output/filename).open("xb") as stream:
                                np.savez_compressed(stream, **retained)
                            frequency_row.update(artifact=filename, artifact_sha256=sha(output/filename),
                                retained_receiver_keys=sorted(retained), no_full_fields_saved=True)
                        persist()
                for name, frequencies in p["bands_hz"].items():
                    if all(f in by_frequency for f in frequencies):
                        vectors = [np.concatenate([by_frequency[f][key] for f in frequencies], axis=0)
                                   for key in ("truth_receiver", "initial_receiver", "tangent_receiver")]
                        row["bands"][name] = dict(status="complete", frequencies_hz=frequencies, **diagnostics(*vectors))
                if row["bands"]["originalband"]["status"] == "complete":
                    inherited, direct = real_dot(gradient, truth-initial), row["bands"]["originalband"]["g_data_dot_delta"]
                    scale = max(abs(inherited), abs(direct), norm(gradient)*norm(truth-initial), 1e-12)
                    error, allowance = abs(inherited-direct), 1e-9*scale
                    row["original_band_saved_gradient_check"] = dict(saved_gradient_dot_delta=inherited,
                        tangent_dot_residual=direct, absolute_error=error, scale=scale, allowance=allowance,
                        numerical_agreement=error <= allowance, formal_enclosure=False)
                row["status"] = "complete" if all(f["status"] == "complete" for f in row["frequencies"]) else "partial_frequency_failure"
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                row.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                persist()
            if source_digest(args.repo) != p["production_source_sha256"]:
                raise ValueError("source changed; remaining cases unattempted")
        for name, digest in p["input_artifacts_sha256"].items():
            if sha(safe(witness, name)) != digest:
                raise ValueError("old witness artifact changed during diagnosis")
        if sha(safe(witness, "complete_receipt.json")) != p["raw_initial_complete_receipt_sha256"]:
            raise ValueError("old complete receipt changed")
        if sha(Path(__file__)) != args.expected_source_sha256 or sha(args.protocol) != args.expected_protocol_sha256:
            raise ValueError("frozen diagnostic source/protocol changed")
        if source_digest(args.repo) != p["production_source_sha256"]:
            raise ValueError("production source changed before completion")
        status["bands_all8"] = {}
        for name in p["bands_hz"]:
            bands = [row["bands"][name] for row in status["cases"]]
            combined = dict(status="unavailable", planned_cases=8,
                            complete_cases=sum(b["status"] == "complete" for b in bands))
            if combined["complete_cases"] == 8:
                data2 = math.fsum(b["data_change_norm"]**2 for b in bands)
                tangent2 = math.fsum(b["tangent_norm"]**2 for b in bands)
                remainder2 = math.fsum(b["remainder_norm"]**2 for b in bands)
                dd = math.fsum(b["g_data_dot_delta"] for b in bands)
                combined.update(status="complete", data_change_norm=math.sqrt(data2), tangent_norm=math.sqrt(tangent2),
                    remainder_norm=math.sqrt(remainder2), remainder_relative_to_data_change=math.sqrt(remainder2/data2) if data2 else None,
                    real_alignment=-dd/math.sqrt(data2*tangent2) if data2*tangent2 else None,
                    g_data_dot_delta=dd, independent_sample_claim=False)
            status["bands_all8"][name] = combined
        numerical = all(row["status"] == "complete" and row.get("original_band_saved_gradient_check", {}).get("numerical_agreement") is True
                        for row in status["cases"])
        status.update(status="complete_diagnostic_attempts", numerical_integrity_pass=numerical,
            cases_complete=sum(r["status"] == "complete" for r in status["cases"]),
            finished_utc=datetime.now(timezone.utc).isoformat(), physical_acceptance=None)
        persist()
        atomic(output/"complete_receipt.json", status)
        return 0 if numerical else 2
    except BaseException as exc:
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}", numerical_integrity_pass=False)
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-probe", action="store_true")
    for key in ("repo", "witness", "output"):
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    p = load_protocol(args.protocol)
    if not args.execute_probe:
        print(json.dumps(dict(status="metadata_only_no_project_arrays_or_solves", source_sha256=sha(Path(__file__)),
            protocol_sha256=sha(args.protocol), planned_cases=8, planned_frequencies=64, planned_rhs_solves=1152)))
        return 0
    if not all((args.repo, args.witness, args.output, args.expected_source_sha256, args.expected_protocol_sha256)):
        parser.error("explicit paths, exact source/protocol hashes and separate root GO required")
    if sha(Path(__file__)) != args.expected_source_sha256 or sha(args.protocol) != args.expected_protocol_sha256:
        raise ValueError("reviewed diagnostic source/protocol hashes changed")
    return execute(args, p)


if __name__ == "__main__":
    raise SystemExit(main())
