"""Three NumPy-only forward/tangent replays; default reads metadata only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping pinned file: " + str(name))
    return path


def load_protocol(path):
    p = json.loads(path.read_bytes())
    base = path.resolve().parent
    fixed = dict(trace_relative_L2=1e-9, gradient_relative_error=1e-3, source_relative_L2=1e-14,
                 direction_relative_L2=1e-14, fold_relative_L2=1e-13, near_zero_relative_normscale=1e-8)
    if p["criteria"] != fixed or p["own_forward_solves"] != 3 or p["direction_checks"] != 6:
        raise ValueError("fixed numerical criteria/full denominator changed")
    original = local_file(base, "original_native_protocol.json")
    if sha(original) != p["original_native_protocol_sha256"]:
        raise ValueError("native protocol changed")
    native = json.loads(original.read_bytes())
    expected_parameters = dict(width=40, dt=.5, nt=2001, source_node=[96,16],
        receiver_nodes=[[x,16] for x in native["geometry"]["receiver_coarse_x_nodes"]])
    if p["inputs"] != native["inputs"] or p["probe_scale"] != .99 or p["directions"] != ["interior", "edge"]:
        raise ValueError("fixed input/model/direction selection changed")
    if p["parameters"] != expected_parameters or p["physical_shape_x_z"] != [192,96] or len(p["inputs"]) != 3:
        raise ValueError("fixed coarse40 grid/time/geometry changed")
    pins = {"numpy_bp_tangent.py": p["core_sha256"], "query_primary_metadata.py": p["metadata_query_source_sha256"],
            "primary_metadata/operator_metadata.json": p["metadata_sha256"]}
    pins.update({"primary_metadata/"+name: h for name,h in p["primary_C_sha256"].items()})
    pins.update(p["copied_primary_python_sha256"])
    pins.update({"canonical_inputs.py":p["canonical_input_source_sha256"],
                 "EXPORT_SOURCE_PLAN.json":p["export_source_plan_sha256"]})
    for name,h in pins.items():
        if sha(local_file(base,name)) != h:
            raise ValueError("source-informed primary/core pin changed: " + name)
    metadata = json.loads((base/"primary_metadata/operator_metadata.json").read_bytes())
    if (any(not r["byte_identical_to_cache"] for r in metadata["records"])
            or not all(r["saved_u_data_unallocated"] for r in metadata["records"] if r["label"] != "rolling_forward")):
        raise ValueError("fresh primary code/allocation authority failed")
    plan = json.loads((base/"EXPORT_SOURCE_PLAN.json").read_bytes())
    if (p.get("schema_version") != 2 or plan["criteria_unchanged"] != fixed
            or [row["case"] for row in plan["cases"]] != p["inputs"]
            or plan["core_sha256"] != p["core_sha256"]
            or plan["native_complete_sha256"] != p["native_complete_sha256"]):
        raise ValueError("V2 exact shared-input/source-plan binding changed")
    return p, native


def scalar_norm(values):
    return math.hypot(*(float(x) for x in values.ravel()))


def relative(a, b):
    numerator, denominator = scalar_norm(a-b), scalar_norm(b)
    good = math.isfinite(numerator) and math.isfinite(denominator)
    return dict(numerator=numerator, denominator=denominator,
                value=numerator/denominator if good and denominator > 0 else None,
                finite=good, zero_denominator=denominator == 0)


def dot(a, b):
    return math.fsum(float(x)*float(y) for x,y in zip(a.ravel(),b.ravel(),strict=True))


def read_npz(path, expected, np):
    if sha(path) != expected:
        raise ValueError("numeric artifact hash changed: " + path.name)
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in archive.files}


def save_npz(path, arrays, np):
    if any(a.ndim > 2 or a.dtype.hasobject for a in arrays.values()):
        raise ValueError("only small numeric2D inputs/traces, never time-by-grid wavefields")
    if sum(a.nbytes for a in arrays.values()) > 16*1024**2:
        raise ValueError("small artifact byte cap exceeded")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def independent_fold(padded, shape, width, np):
    buckets = [[] for _ in range(shape[0]*shape[1])]
    for x in range(padded.shape[0]):
        px = max(0,min(shape[0]-1,x-width))
        for z in range(padded.shape[1]):
            pz = max(0,min(shape[1]-1,z-width))
            buckets[px*shape[1]+pz].append(float(padded[x,z]))
    return np.asarray([math.fsum(values) for values in buckets]).reshape(shape)


def verify_native_input_binding(case, native_run, binding):
    """Recheck exact worker/witness identities after own solves, before comparison."""
    if binding["case"] != case:
        raise ValueError("post-export native case binding differs")
    folder = local_file(native_run, case["id"]+"/status.json").parent
    for name, field in (("status.json", "worker_status_sha256"),
                        ("complete_receipt.json", "worker_complete_sha256")):
        if sha(local_file(folder, name)) != binding[field]:
            raise ValueError("post-export native worker identity changed: " + name)
    status = json.loads((folder/"status.json").read_bytes())
    if status["case"] != case or status["direction_witnesses"] != binding["witnesses"]:
        raise ValueError("post-export direction metadata binding differs")
    for label in ("interior", "edge"):
        witness = binding["witnesses"][label]
        if sha(local_file(folder, witness["artifact"])) != witness["artifact_sha256"]:
            raise ValueError("post-export native direction artifact changed: " + label)
    if sha(local_file(folder, binding["probe_artifact"])) != binding["probe_artifact_sha256"]:
        raise ValueError("post-export native probe input container changed")
    return folder


def compare_case(case, own, own_arrays, native_run, p, native_protocol, expected_calls, np,
                 on_opening, on_opened, on_trace, on_direction, canonical_binding):
    """Called only after ALL three independent forward/tangent solves finish."""
    folder = verify_native_input_binding(case, native_run, canonical_binding)
    status = json.loads((folder/"status.json").read_bytes())
    child_complete = json.loads(local_file(folder,"complete_receipt.json").read_bytes())
    if (status != child_complete or status["case"] != case
            or status["calls"] != [r for r in expected_calls.values() if r["case_id"]==case["id"]]):
        raise ValueError("worker status/completion/case/calls differ from aggregate-pinned receipts")
    before, after = status["source_before"], status["source_after"]
    if (before != after or before["production"] != p["production_source_sha256"]
            or before["driver_sha256"] != p["native_producer_sha256"]
            or before["protocol_sha256"] != p["original_native_protocol_sha256"]
            or before["input_sha256"] != case["sha256"]
            or before["installed"] != native_protocol["source_pins"]
            or before["helper_sha256"] != native_protocol["fixture_helper_sha256"]):
        raise ValueError("native source/input before-after identity mismatch")
    probe_row = next(row for row in status["calls"] if row["call_id"] == "probe_saved")
    adj_row = next(row for row in status["calls"] if row["call_id"] == "probe_adjoint")
    if probe_row["status"] != "complete" or adj_row["status"] != "complete":
        raise ValueError("native probe/gradient unavailable; keep full denominator")
    probe_record, adj_record = probe_row["result"], adj_row["result"]
    for call_id in ["probe_saved","probe_adjoint","coarse40"]:
        recorded = next(row for row in status["calls"] if row["call_id"]==call_id)
        if recorded != expected_calls[case["id"],call_id]:
            raise ValueError("worker call receipt differs from root-pinned aggregate receipt")
    on_opening()
    probe = read_npz(local_file(folder,probe_record["artifact"]),probe_record["artifact_sha256"],np)
    on_opened()
    gradient = read_npz(local_file(folder,adj_record["artifact"]),adj_record["artifact_sha256"],np)
    observed_record = expected_calls[case["id"],"coarse40"]["result"]
    if status["forward_records"]["coarse40"] != observed_record:
        raise ValueError("observed record differs from root-bound coarse40 call result")
    observed = read_npz(local_file(folder,observed_record["artifact"]),observed_record["artifact_sha256"],np)
    times,mask = own_arrays["times_ms"],own_arrays["computed_mask"]
    if (not np.array_equal(times,probe["times_ms"]) or not np.array_equal(mask,probe["computed_mask"])
            or not np.array_equal(times,observed["times_ms"]) or not np.array_equal(mask,observed["computed_mask"])
            or probe["receiver"].shape != (2001,10) or gradient["residual"].shape != (2001,10)):
        raise ValueError("native/own sample axes, receiver shape or computed mask differ")
    expected_source = np.asarray([p["parameters"]["source_node"]],dtype=np.float64)*12.5
    expected_receivers = np.asarray(p["parameters"]["receiver_nodes"],dtype=np.float64)*12.5
    if (not np.array_equal(probe["source_coordinates"],expected_source)
            or not np.array_equal(probe["source_coordinates_after"],expected_source)
            or not np.array_equal(probe["receiver_coordinates"],expected_receivers)
            or not np.array_equal(probe["source_times_ms"],times)
            or not np.array_equal(probe["source_times_after_ms"],times)
            or probe_record["time_bounds"] != own["time_bounds"]
            or adj_record["actual_time_bounds"] != own["time_bounds"]):
        raise ValueError("actual held source/receiver geometry, time or operator bounds differ")
    if (own["physical_velocity_sha256"] != probe_record["model"]["physical_velocity_sha256"]
            or own["physical_m_sha256"] != probe_record["model"]["actual_physical_m_sha256"]):
        raise ValueError("independent and native probe model bytes differ")
    source_check = relative(own_arrays["source"],probe["source_before"])
    if (not np.array_equal(probe["source_before"],probe["source_after"])
            or source_check["value"] is None or source_check["value"] > p["criteria"]["source_relative_L2"]):
        raise ValueError("held excitation mismatch")
    residual = probe["receiver"]-observed["receiver"]
    residual[~mask] = 0
    if not np.array_equal(residual,gradient["residual"]):
        raise ValueError("archived residual does not equal the frozen sampled objective residual")
    trace = relative(own_arrays["receiver"][mask],probe["receiver"][mask])
    trace_pass = trace["value"] is not None and trace["value"] <= p["criteria"]["trace_relative_L2"]
    folded = independent_fold(gradient["padded_gradient"],tuple(p["physical_shape_x_z"]),40,np)
    fold = relative(folded,gradient["production_fold"])
    if (not fold["finite"] or (fold["denominator"]==0 and fold["numerator"]!=0)
            or (fold["value"] is not None and fold["value"] > p["criteria"]["fold_relative_L2"])):
        raise ValueError("archived padded/folded gradient consistency failed")
    on_trace(dict(trace=trace,trace_agreement=trace_pass,source_relative_L2=source_check,
                  padded_fold_relative_L2=fold))
    derivative_records = []
    for label in p["directions"]:
        witness_record = status["direction_witnesses"][label]
        witness = read_npz(local_file(folder,witness_record["artifact"]),witness_record["artifact_sha256"],np)
        direction = own_arrays[label+"_direction"]
        direction_check = relative(direction,witness["direction"])
        if not np.array_equal(own_arrays["physical_probe_m"],witness["probe_m"]):
            hashes = {name:hashlib.sha256(np.ascontiguousarray(value,dtype="<f8").tobytes()).hexdigest()
                for name,value in (("own_context_m",own_arrays["physical_probe_m"]),("witness_context_m",witness["probe_m"]))}
            raise ValueError("exact physical probe context mass mismatch: " + json.dumps(hashes,sort_keys=True))
        if direction_check["value"] is None:
            raise ValueError("physical direction norm comparison unavailable: " + json.dumps(direction_check,sort_keys=True))
        if direction_check["value"] > p["criteria"]["direction_relative_L2"]:
            raise ValueError("physical direction relative-L2 mismatch: " + json.dumps(direction_check,sort_keys=True))
        tangent = own_arrays[label+"_receiver_tangent"]
        tangent_dd = dot(residual[mask],tangent[mask])
        native_dd = dot(gradient["production_fold"],direction)
        padded_dd = dot(gradient["padded_gradient"],np.pad(direction,40,mode="edge"))
        naive_dd = dot(gradient["naive_crop"],direction)
        normscale = scalar_norm(gradient["production_fold"])*scalar_norm(direction)
        informative = normscale > 0 and abs(native_dd) > p["criteria"]["near_zero_relative_normscale"]*normscale
        error = abs(tangent_dd-native_dd)
        denominator = max(abs(tangent_dd),abs(native_dd))
        rel = error/denominator if informative and denominator else None
        derivative_record = dict(direction=label, native_folded_gradient_DD=native_dd,
            independent_tangent_DD=tangent_dd, padded_gradient_DD=padded_dd, naive_crop_DD=naive_dd,
            gradient_direction_normscale=normscale, informative=bool(informative), absolute_error=error,
            relative_error=rel, agreement=bool(rel <= p["criteria"]["gradient_relative_error"]) if rel is not None else None,
            direction_relative_L2=direction_check, native_original_middle_FD_gate=status.get("gradient_numerical_gate"))
        derivative_records.append(derivative_record)
        on_direction(derivative_record)
    return dict(trace=trace, trace_agreement=trace_pass, source_relative_L2=source_check,
        source_bitwise_equal=bool(np.array_equal(own_arrays["source"],probe["source_before"])),
        padded_fold_relative_L2=fold, derivatives=derivative_records,
        native_source_before_after=before==after, native_physical_acceptance=status.get("physical_acceptance"),
        scope="same discrete objective only; no new PDE/objective/gradient calls during comparison")


def execute(args,p):
    import numpy as np

    base=args.protocol.resolve().parent
    spec=importlib.util.spec_from_file_location("_numpy_bp_tangent_pinned_core",base/"numpy_bp_tangent.py")
    core=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    input_spec=importlib.util.spec_from_file_location("_bp_canonical_inputs",base/"canonical_inputs.py")
    input_module=importlib.util.module_from_spec(input_spec)
    input_spec.loader.exec_module(input_module)
    plan=json.loads((base/"EXPORT_SOURCE_PLAN.json").read_bytes())
    input_manifest=input_module.load_manifest(args.inputs.resolve(),args.expected_input_manifest_sha256,
        plan,p["export_source_plan_sha256"],p["canonical_input_source_sha256"])
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(args.native_run.resolve()) or output.is_relative_to(base) or output.is_relative_to(args.inputs.resolve()):
        raise ValueError("fresh output outside native/input/frozen directories required")
    output.mkdir()
    status=dict(status="running_own_solves", role=p["role"], source_sha256=sha(Path(__file__)), protocol_sha256=sha(args.protocol),
        created_utc=datetime.now(timezone.utc).isoformat(), native_solution_output_arrays_opened=False,
        native_derived_numeric_inputs_used=True, native_solution_outputs_used_during_own_solves=False,
        canonical_input_manifest_sha256=args.expected_input_manifest_sha256,
        canonical_input_source_sha256=p["canonical_input_source_sha256"],
        export_source_plan_sha256=p["export_source_plan_sha256"],
        terminal_observation_sha256=sha(args.terminal_observation),
        expected_native_complete_sha256=args.expected_native_complete_sha256,
        cases=[dict(case_id=case["id"],status="unattempted",directions=[dict(direction=d,status="unattempted") for d in p["directions"]]) for case in p["inputs"]],
        own_solves_planned=3,directions_planned=6,physical_acceptance=None)
    atomic_json(output/"status.json",status)
    try:
        # Hash terminal completion before propagation. Shared canonical numeric
        # INPUTS are allowed here; no native solution outputs are decoded.
        if sha(args.native_run/"complete_receipt.json")!=args.expected_native_complete_sha256:
            raise ValueError("native completion/terminal gate failed before own input reads")
        for case,row in zip(p["inputs"],status["cases"],strict=True):
            row["status"]="attempted"
            atomic_json(output/"status.json",status)
            try:
                canonical,input_row=input_module.load_case(args.inputs.resolve(),input_manifest,case,np)
                vp=canonical["physical_probe_velocity"]
                model=canonical["direction_parameter_probe_m"]
                directions=np.asarray([canonical[label+"_direction"] for label in p["directions"]])
                arrays,record=core.propagate(vp,directions,**p["parameters"])
                # The unchanged kernel imports no witness files. Scope its legacy
                # flag explicitly; the wrapper did use native-derived INPUTS.
                record["kernel_opens_native_files"]=record.pop("native_numeric_witnesses_used")
                record.update(native_derived_numeric_inputs_used=True,
                    native_solution_outputs_used_during_own_solve=False,
                    canonical_input_array_sha256=input_row["array_sha256"],
                    canonical_input_artifact_sha256=input_row["artifact_sha256"],
                    direction_context_m_sha256=input_row["array_sha256"]["direction_parameter_probe_m"])
                saved=dict(receiver=arrays["receiver"],times_ms=arrays["times_ms"],computed_mask=arrays["computed_mask"],source=arrays["source"],
                    physical_probe_velocity=vp,physical_probe_m=model,
                    interior_direction=directions[0],edge_direction=directions[1],
                    interior_receiver_tangent=arrays["tangent_receiver"][0],edge_receiver_tangent=arrays["tangent_receiver"][1])
                filename=case["id"]+"_own.npz"
                save_npz(output/filename,saved,np)
                row.update(status="complete_own_solve",artifact=filename,artifact_sha256=sha(output/filename),record=record,
                    canonical_input_artifact_sha256=input_row["artifact_sha256"])
            except Exception as exc:
                row.update(status="failed_own_solve",error=f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                row.update(status="interrupted_own_solve",error=f"{type(exc).__name__}: {exc}",in_flight_state_unknown=True)
                raise
            finally:
                atomic_json(output/"status.json",status)
        if not all(row["status"]=="complete_own_solve" for row in status["cases"]):
            raise RuntimeError("own solves incomplete; native solution outputs remain unopened")
        status.update(status="all_own_solves_complete_before_native_solution_outputs",all3_own_complete_before_native_solution_outputs=True)
        atomic_json(output/"status.json",status)
        if sha(args.native_run/"complete_receipt.json")!=args.expected_native_complete_sha256:
            raise ValueError("explicit completed native run hash mismatch")
        completed=json.loads((args.native_run/"complete_receipt.json").read_bytes())
        if (completed["status"]!="complete_guard_attempts" or completed["source_sha256"]!=p["native_producer_sha256"]
                or completed["protocol_sha256"]!=p["original_native_protocol_sha256"]
                or completed["physical_acceptance"] is not None):
            raise ValueError("native study not complete or wrong source")
        expected_calls={(row["case_id"],row["call_id"]):row for row in completed["calls"]}
        if len(expected_calls)!=57:
            raise ValueError("native57-call denominator differs")
        native_protocol=json.loads((base/"original_native_protocol.json").read_bytes())
        expected_schedule=[]
        for native_case in native_protocol["inputs"]:
            prefix=native_case["id"]
            expected_schedule += [(prefix,v["id"],"forward") for v in native_protocol["forward_calls"]]
            expected_schedule += [(prefix,"probe_saved","forward_saved"),(prefix,"probe_adjoint","adjoint")]
            expected_schedule += [(prefix,f"fd_{d}_{e}_{sign}","forward_fd") for d in p["directions"]
                                  for e in native_protocol["gradient"]["relative_steps"] for sign in ["plus","minus"]]
        if [(r["case_id"],r["call_id"],r["kind"]) for r in completed["calls"]] != expected_schedule:
            raise ValueError("native case/call/kind schedule differs from frozen57-call plan")
        status.update(status="comparing_after_all_own_solves",native_comparison_started=True,
                      native_call_statuses=[{key:r[key] for key in ["case_id","call_id","kind","status"]} for r in completed["calls"]],
                      native_numerical_gate_scope="original57-call/FD outcomes retained separately; this replay cannot replace them")
        atomic_json(output/"status.json",status)
        def on_opening():
            if status["native_solution_output_arrays_opened"] is False:
                status["native_solution_output_arrays_opened"]=None
                atomic_json(output/"status.json",status)
        def on_opened():
            status["native_solution_output_arrays_opened"]=True
            atomic_json(output/"status.json",status)
        for case,row in zip(p["inputs"],status["cases"],strict=True):
            row["comparison_status"]="attempted"
            atomic_json(output/"status.json",status)
            try:
                own_arrays=read_npz(output/row["artifact"],row["artifact_sha256"],np)
                def on_trace(record,row=row):
                    row["comparison_prefix"]=record
                    atomic_json(output/"status.json",status)
                def on_direction(record,row=row):
                    target=next(d for d in row["directions"] if d["direction"]==record["direction"])
                    target.update(status="complete",**record)
                    atomic_json(output/"status.json",status)
                result=compare_case(case,row["record"],own_arrays,args.native_run.resolve(),p,native_protocol,
                    expected_calls,np,on_opening,on_opened,on_trace,on_direction,
                    next(binding for binding in plan["cases"] if binding["case"]==case))
                row.update(comparison_status="complete",comparison=result,directions=[dict(status="complete",**d) for d in result["derivatives"]])
            except Exception as exc:
                row.update(comparison_status="failed",comparison_error=f"{type(exc).__name__}: {exc}")
                for d in row["directions"]:
                    if d["status"]!="complete":
                        d.update(status="unavailable",agreement=None)
            finally:
                atomic_json(output/"status.json",status)
        final_manifest=input_module.load_manifest(args.inputs.resolve(),args.expected_input_manifest_sha256,
            plan,p["export_source_plan_sha256"],p["canonical_input_source_sha256"])
        for case in p["inputs"]:
            input_module.load_case(args.inputs.resolve(),final_manifest,case,np)
        load_protocol(args.protocol)
        if sha(Path(__file__))!=args.expected_source_sha256:
            raise ValueError("runner source changed")
        if sha(args.protocol)!=args.expected_protocol_sha256:
            raise ValueError("protocol bytes changed during replay")
        if sha(args.native_run/"complete_receipt.json")!=args.expected_native_complete_sha256:
            raise ValueError("native completed receipt changed during replay")
        clear=all(row.get("comparison_status")=="complete" and row["comparison"]["trace_agreement"]
                  and all(d.get("agreement") is True for d in row["directions"]) for row in status["cases"])
        status.update(status="complete_comparison",result="ALL_FIXED_NUMERICAL_CHECKS_AGREE" if clear else "MISMATCH_OR_UNINFORMATIVE_RETAINED",
            numerical_agreement_pass=clear,full3_case_6_direction_denominators_retained=True,
            finished_utc=datetime.now(timezone.utc).isoformat(),scope=p["scope"])
        atomic_json(output/"status.json",status)
        atomic_json(output/"complete_receipt.json",status)
        print(json.dumps(dict(status=status["status"],result=status["result"],cases=3,directions=6)))
        return 0 if clear else 2
    except BaseException as exc:
        status.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}",numerical_agreement_pass=False)
        atomic_json(output/"status.json",status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-after-native-terminal",action="store_true")
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--native-run",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--terminal-observation",type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-native-complete-sha256")
    parser.add_argument("--expected-input-manifest-sha256")
    args=parser.parse_args()
    p,_=load_protocol(args.protocol)
    if not args.execute_after_native_terminal:
        print(json.dumps(dict(status="metadata_only_no_BP_or_propagation",source_sha256=sha(Path(__file__)),protocol_sha256=sha(args.protocol),own_solves=3,direction_checks=6)))
        return 0
    if not all([args.inputs,args.native_run,args.output,args.terminal_observation,args.expected_source_sha256,args.expected_protocol_sha256,args.expected_native_complete_sha256,args.expected_input_manifest_sha256]):
        parser.error("exact hashes, native terminal observation, inputs/native/new output required")
    if sha(Path(__file__))!=args.expected_source_sha256 or sha(args.protocol)!=args.expected_protocol_sha256:
        raise ValueError("runner/protocol freeze changed")
    if args.expected_native_complete_sha256 != p["native_complete_sha256"]:
        raise ValueError("native completion does not match V2 source-plan binding")
    if not args.terminal_observation.is_file():
        raise ValueError("root terminal observation required")
    return execute(args,p)


if __name__=="__main__":
    raise SystemExit(main())
