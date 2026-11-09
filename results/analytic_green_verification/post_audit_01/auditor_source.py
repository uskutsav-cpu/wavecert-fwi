"""Independent z^2-substitution Green audit; no production import or PDE solve."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad

COMPLETE_SHA = "91281a563841fea5d34359ebd7400e9cbe7036387d2321816edb5d8ec9412b04"
PROTOCOL_SHA = "729e23e1a8a5373ba5492a47f2259d0bcb9a0618b25ddc87f3d73e0ea44337e5"
RUNNER_SHA = "ba7b03027d837f6e834e47ebf13c79768287d6d2963b1df3b898a6cbce822b69"
SOURCE_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON: "+value)
    return json.loads(path.read_bytes(), parse_constant=invalid)


def norm(values):
    return math.hypot(*(float(x) for x in np.asarray(values).ravel()))


def ratio(error, reference):
    numerator, denominator = norm(error), norm(reference)
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator/denominator if denominator else None,
            "zero_denominator": denominator == 0}


def metrics(discrete, analytic, times, valid):
    early, late = valid & (times <= 250.), valid & (times > 250.)
    error = discrete-analytic
    return {"full_relative": ratio(error[valid], analytic[valid]),
            "early_relative": ratio(error[early], analytic[early]),
            "late_mismatch_over_full_analytic": ratio(error[late], analytic[valid]),
            "sample_counts": {"full": int(valid.sum()), "early": int(early.sum()), "late": int(late.sum())},
            "physical_quality_pass": None}


def bundle_snapshot(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}


def source_guard(repo):
    files = [repo/"pyproject.toml"]+list((repo/"src").rglob("*.py"))
    files += [p for p in (repo/"configs").rglob("*") if p.is_file()]
    files += [p for p in (repo/"scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    files += list((repo/".github/workflows").glob("*.yml"))+list(repo.glob("requirements*.txt"))
    digest, count = hashlib.sha256(), 0
    for path in sorted(set(files)):
        if path.is_file():
            digest.update(path.relative_to(repo).as_posix().encode()+b"\0"+sha(path).encode()+b"\0")
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "bundle", "original", "run", "output", "criteria"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-criteria-sha256", required=True)
    args = parser.parse_args()
    repo, bundle, original, run, output, criteria_path = [getattr(args, name).resolve()
        for name in ("repo", "bundle", "original", "run", "output", "criteria")]
    if sha(Path(__file__)) != args.expected_source_sha256 or sha(criteria_path) != args.expected_criteria_sha256:
        parser.error("pre-run frozen auditor/criteria SHA differs")
    if output.exists() or any(output.is_relative_to(p) for p in (repo, bundle, original, run)):
        parser.error("fresh OUTER output directory required")
    criteria = read(criteria_path)
    if (criteria["quad_epsabs"] != 1e-11 or criteria["quad_epsrel"] != 1e-11
            or criteria["full_valid_curve_relative_L2_maximum"] != 1e-10
            or criteria["valid_indices_inclusive"] != [1, 1099]):
        parser.error("fixed root-authorized post-audit criteria differ")
    output.mkdir()
    (output/"auditor_source.py").write_bytes(Path(__file__).read_bytes())
    (output/"criteria.json").write_bytes(criteria_path.read_bytes())
    report = {"role": "Independent AI post-outcome numerical audit, not student interpretation",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": args.expected_source_sha256, "criteria_sha256": args.expected_criteria_sha256,
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
              "PDE_or_production_import_or_original_write": False,
              "criteria": criteria, "orders": [], "cases": [], "quad_messages": []}
    issues = []
    checks = 0
    alternate = np.zeros(1101); alternate_step = np.zeros(1101)
    estimated_errors = np.zeros(1101); estimated_step_errors = np.zeros(1101)
    evaluated = np.zeros(1101, dtype=bool)

    def check(condition, context):
        nonlocal checks
        checks += 1
        if not bool(condition):
            issues.append(context)

    def near(actual, expected, context):
        if actual is None or expected is None:
            check(actual is expected, context)
        else:
            check(math.isfinite(actual) and math.isfinite(expected)
                  and abs(actual-expected) <= criteria["scalar_atol"]+criteria["scalar_rtol"]*max(abs(actual), abs(expected)), context)

    def metrics_near(actual, expected, context):
        check(actual["sample_counts"] == expected["sample_counts"], context+"/counts")
        check(expected["physical_quality_pass"] is None, context+"/no physical pass")
        for metric in ("full_relative", "early_relative", "late_mismatch_over_full_analytic"):
            for key in ("numerator", "denominator", "value"):
                near(actual[metric][key], expected[metric][key], context+"/"+metric+"/"+key)
            check(actual[metric]["zero_denominator"] == expected[metric]["zero_denominator"], context+"/zero denominator")

    try:
        check(sha(run/"complete_receipt.json") == COMPLETE_SHA, "root terminal complete receipt pin")
        status, complete = read(run/"status.json"), read(run/"complete_receipt.json")
        check(status == complete and status["status"] == "complete_analytical_diagnostic", "terminal status/receipt")
        check(status["source_before"] == status["source_after"] and status["original_before"] == status["original_after"], "producer input/source invariance")
        check(status["source_and_original_unchanged"] and status["internal_numeric_gates_pass"] and status["all_input_software_gates_pass"], "producer complete gates")
        check(sha(bundle/"compare_green.py") == RUNNER_SHA and sha(bundle/"protocol.json") == PROTOCOL_SHA, "fixed producer source and protocol")
        protocol = read(bundle/"protocol.json")
        before_bundle = bundle_snapshot(bundle)
        check(before_bundle == status["source_before"], "actual bundle matches full producer manifest")
        before_old = {name: sha(original/name) for name in {**protocol["old_run_metadata_pins"], **protocol["old_raw_pins"]}}
        check(before_old == status["original_before"] == {**protocol["old_run_metadata_pins"], **protocol["old_raw_pins"]}, "all old artifacts pinned")
        guard_before = source_guard(repo)
        check(guard_before["sha256"] == SOURCE_SHA, "actual production source9f")
        check(len(status["orders"]) == 3 and [r["order"] for r in status["orders"]] == [128,256,512], "all3 order denominator")
        check(len(status["cases"]) == 7 and [r["case_id"] for r in status["cases"]] == protocol["case_ids"], "all7 case denominator")
        check(all(r["status"] == "complete" for r in status["orders"]+status["cases"]), "no omitted failed/unattempted orders/cases")
        times = np.arange(1101, dtype=np.float64)
        valid = (times >= 1) & (times <= 1099)
        travel, strength = 25., 100.

        def ricker(time):
            if time < 0:
                return 0.
            x = math.pi*.005*(time-200.)
            return (1.-2.*x*x)*math.exp(-x*x)

        source = np.array([ricker(float(t)) for t in times])
        step_exact = np.array([0. if t <= travel else strength/(2*math.pi)*math.acosh(t/travel) for t in times])
        total_neval = 0
        # Different substitution and scalar math.exp, adaptive Gauss-Kronrod
        # QUADPACK; no cosh map, GL nodes, saved curves or traces inform this.
        for index, time in enumerate(times):
            if time > travel:
                upper = math.sqrt(time-travel)
                for is_step in (False, True):
                    def integrand(z):
                        forcing = 1. if is_step else ricker(time-travel-z*z)
                        return forcing/math.sqrt(2*travel+z*z)
                    answer = quad(integrand, 0., upper, epsabs=criteria["quad_epsabs"],
                                  epsrel=criteria["quad_epsrel"], limit=criteria["quad_limit"], full_output=1)
                    value, error, info = answer[:3]
                    total_neval += int(info["neval"])
                    if len(answer) > 3:
                        report["quad_messages"].append({"index": index, "step_forcing": is_step, "message": str(answer[3])})
                        check(False, f"quad convergence message at{index}/step{is_step}")
                    scale = strength/math.pi
                    if is_step:
                        alternate_step[index], estimated_step_errors[index] = scale*value, scale*error
                    else:
                        alternate[index], estimated_errors[index] = scale*value, scale*error
            evaluated[index] = True
        save = output/"alternate_z_squared_curve.npz"
        np.savez_compressed(save, times_ms=times, valid_mask=valid, causal_source=source, analytic=alternate,
                            unit_step=alternate_step, step_closed_form=step_exact, computed_sample_mask=evaluated,
                            quad_estimated_absolute_error=estimated_errors,
                            quad_step_estimated_absolute_error=estimated_step_errors)
        check(bool(np.all(evaluated)) and bool(np.all(np.isfinite(alternate))), "all1101 alternate samples finite/evaluated")
        step_error = max(abs(float(x-y)) for x, y in zip(alternate_step, step_exact, strict=True))/(1+max(abs(float(x)) for x in step_exact))
        check(step_error <= criteria["step_normalized_max_abs_maximum"], "alternate step acosh control")
        report.update(alternate_curve_sha256=sha(save), quad_function_evaluations=total_neval,
                      alternate_step_normalized_max_abs_error=step_error,
                      maximum_quad_estimated_absolute_error=float(np.max(estimated_errors)),
                      quad_error_estimate_scope="QUADPACK estimate only, not validated enclosure")
        gl = {}
        for order_row in status["orders"]:
            order = order_row["order"]
            path = run/order_row["artifact"]
            check(sha(path) == order_row["artifact_sha256"], f"GL{order}/artifact SHA")
            with np.load(path, allow_pickle=False) as saved:
                values = {k: saved[k].copy() for k in saved.files}
            check(set(values) == {"times_ms","source","valid_mask","analytic","step_exact","step_quadrature","quadrature_nodes","quadrature_weights"}, f"GL{order}/all saved fields")
            check(np.array_equal(values["times_ms"], times) and np.array_equal(values["valid_mask"], valid), f"GL{order}/times and mask")
            excitation = ratio(values["source"]-source, source)
            check(excitation["value"] <= criteria["source_relative_L2_maximum"], f"GL{order}/independent scalar-exp source")
            nodes, weights = values["quadrature_nodes"], values["quadrature_weights"]
            moments = [abs(math.fsum(float(w)*float(x)**k for x,w in zip(nodes,weights,strict=True))-(0. if k%2 else 2/(k+1))) for k in range(7)]
            check(nodes.shape == weights.shape == (order,) and bool(np.all(weights>0)) and bool(np.all(np.diff(nodes)>0))
                  and bool(np.all(np.abs(nodes)<1)) and max(moments)<=1e-12, f"GL{order}/nodes weights moments")
            for computed, recorded in zip(moments, order_row["moment_absolute_errors"], strict=True):
                near(computed, recorded, f"GL{order}/moment residual receipt")
            recomputed_step_error = max(abs(float(x-y)) for x,y in zip(values["step_quadrature"],step_exact,strict=True))/(1+max(abs(float(x)) for x in step_exact))
            check(recomputed_step_error <= 1e-12, f"GL{order}/saved step control")
            check(bool(np.all(np.isfinite(values["step_exact"])))
                  and max(abs(float(x-y)) for x,y in zip(values["step_exact"],step_exact,strict=True))
                  <= criteria["scalar_atol"]+criteria["scalar_rtol"]*max(abs(float(x)) for x in step_exact),
                  f"GL{order}/saved closed-form step array")
            near(recomputed_step_error,order_row["step_control_error"],f"GL{order}/step receipt")
            curve_error = ratio(values["analytic"][valid]-alternate[valid],alternate[valid])
            check(curve_error["value"] <= criteria["full_valid_curve_relative_L2_maximum"],f"GL{order}/alternate quadrature agreement")
            report["orders"].append({"order":order,"alternate_curve_relative_L2":curve_error,
                "source_relative_L2":excitation,"scalar_moment_errors":moments,"step_error":recomputed_step_error})
            gl[order]=values["analytic"]
        for recorded in status["quadrature_convergence"]:
            lower,upper=recorded["lower_order"],recorded["upper_order"]
            actual=ratio((gl[lower]-gl[upper])[valid],gl[upper][valid])
            for key in ("numerator","denominator","value"):
                near(actual[key],recorded[key],"GL convergence/"+key)
            check(actual["value"] <= 1e-10 and recorded["gate"],"both original internal convergence gates")
        # Original arrays are decoded only after the alternate full curve and
        # controls are saved; original compressed file SHA reads preceded it.
        for row in status["cases"]:
            case=row["case_id"]
            path=run/row["artifact"]
            check(sha(path)==row["artifact_sha256"],case+"/comparison artifact SHA")
            with np.load(path,allow_pickle=False) as saved:
                comp={k:saved[k].copy() for k in saved.files}
            with np.load(original/(case+"_raw.npz"),allow_pickle=False) as saved:
                old={k:saved[k].copy() for k in saved.files}
            check(np.array_equal(comp["times_ms"],times) and np.array_equal(comp["valid_mask"],valid),case+"/saved mask/time")
            check(np.array_equal(comp["receiver"],old["receiver"]) and np.array_equal(comp["source"],old["source_before"]),case+"/exact old raw binding")
            check(np.array_equal(comp["analytic512"],gl[512]) and np.array_equal(comp["difference"],old["receiver"][:,0]-gl[512]),case+"/saved comparison binding")
            check(all(np.array_equal(old[k],times) for k in ("times_before_ms","times_after_ms","source_times_before_ms","source_times_after_ms","receiver_times_ms")),case+"/all original time axes")
            check(np.array_equal(old["source_before"],old["source_after"]) and np.array_equal(old["source_coordinates_before_m"],old["source_coordinates_after_m"]),case+"/unchanged old source")
            distance=norm(old["receiver_coordinates_m"]-old["source_coordinates_before_m"])
            check(distance==50. and old["receiver"].shape==(1101,1) and old["receiver"][0,0]==old["receiver"][-1,0]==0,case+"/geometry and uncomputed sentinels")
            source_error=ratio(old["source_before"][:,0]-source,source)
            check(source_error["value"]<=1e-14,case+"/same scalar-exp excitation")
            primary_metrics=metrics(old["receiver"][:,0],gl[512],times,valid)
            alternate_metrics=metrics(old["receiver"][:,0],alternate,times,valid)
            metrics_near(primary_metrics,row["metrics"],case+"/fsum-hypot receipt metrics")
            report["cases"].append({"case_id":case,"primary_case":case=="reference_321",
                 "recomputed_GL512_metrics":primary_metrics,"alternate_quad_metrics":alternate_metrics,
                 "source_relative_L2":source_error,"physical_quality_pass":None})
        report.update(actual_source_before=guard_before,actual_source_after=source_guard(repo))
        check(report["actual_source_after"]==guard_before,"actual source9f unchanged after audit")
        check(bundle_snapshot(bundle)==before_bundle,"full original source bundle unchanged after audit")
        check({name:sha(original/name) for name in before_old}==before_old,"all original seven-case files unchanged after audit")
        check(read(run/"status.json")==status and sha(run/"complete_receipt.json")==COMPLETE_SHA,"analytical metadata unchanged after audit")
        check(all(sha(run/r["artifact"])==r["artifact_sha256"] for r in status["orders"]+status["cases"]),"all10 analytical NPZ files unchanged after audit")
    except BaseException as exc:
        issues.append(type(exc).__name__+": "+str(exc))
        report["exception_traceback"]=traceback.format_exc()
        if not (output/"alternate_z_squared_curve.npz").exists():
            np.savez_compressed(output/"partial_alternate_curve.npz",analytic=alternate,step=alternate_step,computed_sample_mask=evaluated)
    report.update(result="CLEAR_INDEPENDENT_GREEN_POSTAUDIT" if not issues else "ISSUES_REQUIRE_REVIEW",
                  checks=checks,issues=issues,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_formula="s=a+z^2: A/pi integral q(t-a-z^2)/sqrt(2a+z^2) dz",
                  physical_quality_pass=None,
                  limits="Post-outcome numerical crosscheck only. No formal quadrature/FP enclosure, no newPDE, no damping/adjoint/inversion approval, no seven independent environments or student design.")
    report["artifacts"]={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    with (output/"receipt.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"result":report["result"],"checks":checks,"issues":len(issues),"output":str(output),"receipt_sha256":sha(output/"receipt.json")}))
    return 0 if not issues else 1


if __name__=="__main__":
    raise SystemExit(main())
