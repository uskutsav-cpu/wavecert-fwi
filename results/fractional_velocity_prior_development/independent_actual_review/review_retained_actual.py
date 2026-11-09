"""Independent saved-byte/scalar reproduction; no producer/auditor/PDE imports."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import struct
import zipfile

import numpy as np

BASE = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RAW = BASE / "work/prior48_remote_study_01"
EX = RAW / "extracted"
RUN = EX / "study"
AUDIT = BASE / "work/prior48_independent_audit_01/report.json"
CRITERIA = BASE / "work/new-phase-staging/fractional_velocity_prior48_post_audit/criteria.json"
INPUT = BASE / "work/wavecert-fwi/data/reference/subsurfacegen_parent_split/validation.npz"
ARMS = ("normalized_exact_data_only", "normalized_exact_fractional_update_prior")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def model_sha(v):
    return hashlib.sha256(b"".join(struct.pack("<d", x) for x in v)).hexdigest()


class Checks:
    def __init__(self):
        self.checks = 0
        self.numeric = 0
        self.failures = []

    def require(self, value, label):
        self.checks += 1
        if not value:
            self.failures.append(label)

    def near(self, a, b, label, rtol, atol):
        self.numeric += 1
        self.require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol+rtol*abs(a), label)


def scalar_truth(coarse):
    velocity = [[1/math.sqrt(float(m)) for m in row] for row in coarse]
    fine = []
    for z in range(49):
        qz = (z-1)/2
        loz = math.floor(qz)
        wz = qz-loz
        iz, jz = max(0, min(23, loz)), max(0, min(23, loz+1))
        line = []
        for x in range(49):
            qx = (x-1)/2
            lox = math.floor(qx)
            wx = qx-lox
            ix, jx = max(0, min(23, lox)), max(0, min(23, lox+1))
            a = velocity[iz][ix] + wx*(velocity[iz][jx]-velocity[iz][ix])
            b = velocity[jz][ix] + wx*(velocity[jz][jx]-velocity[jz][ix])
            line.append(a + wz*(b-a))
        fine.append(line)
    return [1/(v*v) for row in fine for v in row], fine


def scalar_initial(fine):
    # Independent separable Gaussian, sigma3.2/truncate4/half-sample reflect.
    weights = [math.exp(-.5*(k/3.2)**2) for k in range(-13, 14)]
    total = math.fsum(weights)
    weights = [w/total for w in weights]

    def reflected(i):
        return -i-1 if i < 0 else 97-i if i >= 49 else i

    first = [[math.fsum(w*fine[reflected(z+k)][x] for k, w in zip(range(-13, 14), weights))
              for x in range(49)] for z in range(49)]
    second = [[math.fsum(w*first[z][reflected(x+k)] for k, w in zip(range(-13, 14), weights))
               for x in range(49)] for z in range(49)]
    return [1/(v*v) for row in second for v in row]


def quality(model, truth):
    tv = [1/math.sqrt(x) for x in truth]
    return {
        "relative_squared_slowness_L2": math.sqrt(math.fsum((x-y)**2 for x, y in zip(model, truth)) / math.fsum(x*x for x in truth)),
        "relative_physical_velocity_L2": math.sqrt(math.fsum((1/math.sqrt(x)-y)**2 for x, y in zip(model, tv)) / math.fsum(x*x for x in tv)),
    }


def regularizer(model, initial):
    q = [math.sqrt(a)/math.sqrt(b)-1 for a, b in zip(initial, model)]
    edges = []
    for z in range(49):
        for x in range(49):
            i = z*49+x
            if x < 48:
                edges.append((q[i]-q[i+1])**2)
            if z < 48:
                edges.append((q[i]-q[i+49])**2)
    return (math.fsum(x*x for x in q)+16*math.fsum(edges))/(2*2401)


def execute():
    checks = Checks()
    expected = {
        RAW / "artifact.zip": "7d3d536d164e5a4c1ad904da09b18e051db519663d7512911b217c50df0085be",
        RAW / "ROOT_TERMINAL_OBSERVATION.json": "bd255c3b834d52c5c2d377bc85c08e4d49cdae649af81831df64a766703f24de",
        RUN / "complete_receipt.json": "ca69e5e36b2c18727af9d45de84a9d2c37310ceb4d8e1c7069898f4c2b04de85",
        AUDIT: "c3b528d942cba432b38be6bd2f46139402b058687e9e220781e10d57a4161baa",
        INPUT: "36d11aba94ae9093cac888e9b28a2d01edcef1aa3a54b12f8e10269b963db2fc",
        CRITERIA: "b06e9618b4d980ed0aa08c05a0da136d0ba3e37ecc422be4df411a5f3b96edf6",
    }
    for path, value in expected.items():
        checks.require(sha(path) == value, "bound file: "+str(path.relative_to(BASE)))
    if checks.failures:
        raise ValueError(checks.failures)
    c, status, complete, audit = map(read, [CRITERIA, RUN/"status.json", RUN/"complete_receipt.json", AUDIT])
    terminal, child, supervisor = map(read, [RAW/"ROOT_TERMINAL_OBSERVATION.json", EX/"supervision/child_exit.json", EX/"supervision/supervisor_exit.json"])
    inv = read(EX/"UPLOAD_INVENTORY.json")
    checks.require(inv["complete_evidence_collection"] is True and inv["omitted"] == [], "complete artifact collection")
    entries = inv["retained"]
    checks.require(len(entries) == len({v["path"] for v in entries}) == 131, "full unique131 inventory")
    for entry in entries:
        path = (EX/entry["path"]).resolve()
        checks.require(path.is_relative_to(EX.resolve()), "safe inventory path")
        checks.require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "inventory bytes: "+entry["path"])
        if "copy_and_source_stable" in entry:
            checks.require(entry["copy_and_source_stable"] is True and entry["sha256"] == entry["source_snapshot_sha256"], "copy stable: "+entry["path"])
    checks.require(sum(v["bytes"] for v in entries) == inv["retained_bytes"], "inventory total bytes")
    with zipfile.ZipFile(RAW/"artifact.zip") as archive:
        checks.require(archive.testzip() is None, "ZIP member CRCs")
        for name in archive.namelist():
            checks.require((EX/name).resolve().is_relative_to(EX.resolve()) and archive.read(name) == (EX/name).read_bytes(), "ZIP extracted bytes: "+name)
    checks.require(terminal["status_sha256"] == sha(RUN/"status.json"), "root status hash")
    checks.require(terminal["remote_child_exit_receipt_sha256"] == sha(EX/"supervision/child_exit.json"), "root child hash")
    checks.require(type(child["actual_returncode"]) is int and child["actual_returncode"] == terminal["actual_exit_code"] == supervisor["actual_child_returncode"] == 0, "observed actual child exit0")
    checks.require(child["child_PID"] == supervisor["child_PID"] and child["complete_receipt_sha256"] == terminal["complete_receipt_sha256"], "child completion identity")
    checks.require({k:v for k,v in complete.items() if k != "artifact_sha256"} == status, "status complete identity")
    checks.require(set(complete["artifact_sha256"]) == {p.name for p in RUN.iterdir() if p.is_file() and p.name != "complete_receipt.json"}, "full study artifact denominator")
    for name, value in complete["artifact_sha256"].items():
        checks.require(sha(RUN/name) == value, "study artifact: "+name)
    p = read(RUN/"protocol.json")
    checks.require(sha(RUN/"driver_source.py") == c["driver_sha256"] and sha(RUN/"protocol.json") == c["protocol_sha256"] and sha(RUN/"FROZEN48_SCHEDULE.json") == c["schedule_sha256"], "exact frozen producer source")
    checks.require(status["source_before"] == status["source_after"], "unchanged producer source")
    checks.require(status["source_before"]["production"]["repository_source_sha256"] == c["production_sha256"], "native production9f")
    checks.require(audit["checks"] == 51613 and audit["numeric_comparisons"] == 1040 and audit["issues"] == [], "actual audit denominator")
    checks.require(audit["verified_full_study_credit"] is True and audit["scientific_goals_complete"] is False, "scientific claim limits")
    with np.load(INPUT, allow_pickle=False) as decoded:
        raw = decoded["models"]
        checks.require(raw.shape == (192,24,24) and raw.dtype == np.float32, "frozen raw shape/dtype")
        truth, initial = [], []
        for identity in p["identities"]:
            t, v = scalar_truth(raw[identity["array_index"]])
            truth.append(t)
            initial.append(scalar_initial(v))
    rng, scheduled = random.Random(2026110901), []
    for rep in range(3):
        for case in range(8):
            arms = [(ARMS[0],0.0),(ARMS[1],100.0)]
            rng.shuffle(arms)
            for position,(arm,weight) in enumerate(arms):
                scheduled.append((len(scheduled),case,rep,position,arm,weight))
    checks.require(len(status["runs"]) == len(audit["rows"]) == 48, "all48 rows")
    rows = []
    for row, reviewed, order in zip(status["runs"],audit["rows"],scheduled):
        index, case, rep, position, arm, weight = order
        label = f"run{index}"
        checks.require((row["run_index"],row["case_index"],row["repetition"],row["order_position"],row["method"],row["weight"]) == order, label+" fixed slot")
        checks.require(row["status"] == "complete" and row["final_study_eligible"] is True and reviewed["eligible"] is True, label+" complete eligible")
        payload, graph = read(RUN/row["return_artifact"]), read(RUN/row["audit_artifact"])
        t = row["timing"]
        checks.require(t["actual_complete_return_seconds"] <= 8 and 0 <= t["inner_return_age"] <= t["actual_complete_return_seconds"] and t["eligible"] is True, label+" strict caller clock")
        checks.require(t["actual_complete_return_seconds"] == reviewed["wall"] and payload["method_valid"] is True and payload["weight"] == weight, label+" reviewed clock/method")
        points = [q for q in payload["engine"]["checkpoints"] if q["within_budget"] and q["elapsed_seconds"] <= 7.5]
        checks.require(bool(points) and payload["engine"]["budget_seconds"] == 7.5 and payload["engine"]["costs_complete"] is True, label+" strict engine clock")
        models = {k:payload[k] for k in ("initial_model","final_model")}
        for name, model in models.items():
            checks.require(len(model) == 2401 and all(math.isfinite(x) and 1/36 <= x <= 1 for x in model), label+" model domain "+name)
            q = quality(model, truth[case])
            for key, value in q.items():
                for stored in [graph["records"][name][key], reviewed["initial_quality" if name == "initial_model" else "final_quality"][key], row["initial_quality" if name == "initial_model" else "primary_quality"][key]]:
                    checks.near(value,stored,label+" scalar quality "+name+"/"+key,c["quality_rtol"],c["quality_atol"])
        checks.require(all(abs(a-b) <= c["initial_model_atol"]+c["initial_model_rtol"]*abs(b) for a,b in zip(models["initial_model"],initial[case])),label+" independent scalar Gaussian initializer")
        final_sha = model_sha(models["final_model"])
        checks.require(points[-1]["model_sha256"] == final_sha == payload["final_raw_native_header"]["model_sha256"],label+" last eligible final identity")
        raw0 = payload["initial_native_header"]["native_data_receipt_header"]["objective"]
        rawfinal = payload["final_raw_native_header"]["objective"]
        ratio = rawfinal/raw0
        R = regularizer(models["final_model"],models["initial_model"]) if weight else 0.0
        if weight:
            checks.near(R,payload["final_R"],label+" scalar R",c["prior_rtol"],c["prior_atol"])
        for stored in [payload["final_F"],payload["engine"]["final_objective"],payload["engine"]["adapter"]["current_accepted"]["objective"]]:
            checks.near(ratio+weight*R,stored,label+" scalar F",c["composed_F_rtol"],c["composed_F_atol"])
        for stored in [row["common_raw_Phi_ratio"],reviewed["common_raw_Phi_ratio"],payload["common_raw_Phi_ratio"]]:
            checks.near(ratio,stored,label+" raw ratio",c["objective_ratio_rtol"],c["objective_ratio_atol"])
        native, prep = payload["final_native_owner_cumulative_costs"],row["preparation_costs"]
        o, ph = native["oracle"],native["physics"]
        checks.require(prep["factorizations"] == 8 and prep["forward_rhs"] == 48,label+" repeated paid preparation")
        checks.require(ph["factorizations"] == 8*o["objective_evaluations"] and ph["forward_rhs"] == 6*ph["factorizations"] and ph["adjoint_rhs"] == 48*o["gradient_evaluations"],label+" native LU/RHS count identities")
        checks.require(native == reviewed["final_native_snapshot"] and prep == reviewed["preparation_counts"],label+" single native/preparation snapshot")
        a, b = quality(models["initial_model"],truth[case]), quality(models["final_model"],truth[case])
        vi = [1/math.sqrt(x) for x in models["initial_model"]]
        displacement = math.sqrt(math.fsum((1/math.sqrt(x)-v)**2 for x,v in zip(models["final_model"],vi))/math.fsum(v*v for v in vi))
        rows.append(dict(run_index=index,case=case,rep=rep,arm=arm,initial_error=a["relative_physical_velocity_L2"],final_error=b["relative_physical_velocity_L2"],raw_ratio=ratio,wall=t["actual_complete_return_seconds"],relative_velocity_displacement_from_initial=displacement,paid_factorizations=prep["factorizations"]+ph["factorizations"],paid_forward_rhs=prep["forward_rhs"]+ph["forward_rhs"],paid_adjoint_rhs=ph["adjoint_rhs"],stored_postgraph_factorizations=graph["factorizations"],stored_postgraph_rhs=graph["forward_rhs"]))
    cases = []
    for case in range(8):
        groups = {arm:[r for r in rows if r["case"] == case and r["arm"] == arm] for arm in ARMS}
        checks.require(all(len(v) == 3 and {r["rep"] for r in v} == {0,1,2} for v in groups.values()),f"case{case} complete paired groups")
        med = {arm:{key:statistics.median(r[key] for r in group) for key in ["initial_error","final_error","raw_ratio","wall","relative_velocity_displacement_from_initial"]} for arm,group in groups.items()}
        a,b = med[ARMS[0]],med[ARMS[1]]
        success = b["final_error"] <= .9*a["initial_error"] and b["final_error"] <= .9*a["final_error"] and b["raw_ratio"] <= .75
        harm = b["final_error"] <= 1.1*a["initial_error"]
        saved = audit["summary"]["cases"][case]
        checks.require(saved["case_index"] == case and saved["successful_case"] is success and saved["harm_gate_pass"] is harm,f"case{case} fixed gates")
        for key,value in [("initial_error",a["initial_error"]),("baseline_error",a["final_error"]),("candidate_error",b["final_error"]),("candidate_raw_Phi_ratio",b["raw_ratio"])]:
            checks.near(value,saved[key],f"case{case} median {key}",c["quality_rtol"],c["quality_atol"])
        cases.append(dict(case=case,initial_error=a["initial_error"],baseline_error=a["final_error"],prior_error=b["final_error"],baseline_over_initial=a["final_error"]/a["initial_error"],prior_over_initial=b["final_error"]/a["initial_error"],prior_over_baseline=b["final_error"]/a["final_error"],prior_raw_ratio=b["raw_ratio"],successful_case=success,harm_gate_pass=harm,baseline_wall=a["wall"],prior_wall=b["wall"],prior_velocity_displacement=b["relative_velocity_displacement_from_initial"]))
    gate = sum(v["successful_case"] for v in cases) >= 6 and all(v["harm_gate_pass"] for v in cases)
    checks.require(gate is False and audit["summary"]["proposed_development_continuation_gate_pass"] is gate,"global recovery gate failed")
    arm_stats = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        arm_stats[arm] = dict(runs=len(group),wall_min=min(r["wall"] for r in group),wall_max=max(r["wall"] for r in group),wall_median=statistics.median(r["wall"] for r in group),totals={key:sum(r[key] for r in group) for key in ["paid_factorizations","paid_forward_rhs","paid_adjoint_rhs","stored_postgraph_factorizations","stored_postgraph_rhs"]})
    result = dict(status="CLEAR" if not checks.failures else "ISSUES",checks=checks.checks,numeric_comparisons=checks.numeric,issues=checks.failures,source_sha256=sha(Path(__file__)),bound_files={str(k.relative_to(BASE)):sha(k) for k in expected},inventory_files=131,full_runs=48,cases=cases,arm_statistics=arm_stats,wall_range=[min(r["wall"] for r in rows),max(r["wall"] for r in rows)],prior_beats_baseline_cases=sum(r["prior_error"] < r["baseline_error"] for r in cases),prior_worse_than_initial_cases=sum(r["prior_over_initial"] > 1 for r in cases),baseline_worse_than_initial_cases=sum(r["baseline_over_initial"] > 1 for r in cases),successful_cases=sum(r["successful_case"] for r in cases),all_harm_gates_pass=all(r["harm_gate_pass"] for r in cases),fixed_continuation_gate_pass=gate,rows=rows,scope="Source-aware independent AI retained-byte/scalar reproduction. Independently decoded exact frozen localNPZ; scalar half-index truth and separable Gaussian initializer, sqrt/fsum errors andR/F. No producer/auditor imports or nativephysics. Stored rawPhi/observation/solve evidence is not independently PDE-regenerated. No newgeology/human/studentownership or fullscientificgoal proof.")
    out = HERE/"actual_review_report.json"
    with out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["rows","cases","bound_files"]},indent=2))
    return 0 if not checks.failures else 2


if __name__ == "__main__":
    raise SystemExit(execute())
