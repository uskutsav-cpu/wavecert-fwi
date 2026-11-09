"""Held stdlib serial parent: exactly two source-bound case0 pilot slots.

Default metadata only. Real execution requires separate reviewed source/runtime,
control evidence, a new auditor and exact root GO. No numerical imports here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTHORIZATION = "GO_TRF_CASE0_TWO_RETURN_INTEGRATION_PILOT"


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value)+b"\n");stream.flush();os.fsync(stream.fileno())


def replace_own(path,value):
    path = Path(path);temporary = path.with_name(path.name+".atomic-tmp")
    write_new(temporary,value)
    os.replace(temporary,path)


def parent_contract():
    return {name:digest(HERE/name) for name in ("serial_pilot.py","PILOT_PLAN.json","test_parent.py")}


def bounded_json(path,maximum=16*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("nonregular or over-cap JSON: "+str(path))
    raw = path.read_bytes()
    if len(raw)>maximum:raise ValueError("JSON grew beyond fixed cap")
    return json.loads(raw)


def bound_file(base,row):
    relative = row.get("path")
    if not isinstance(relative,str) or Path(relative).is_absolute():raise ValueError("relative bound-file path required")
    original = base/relative;path = original.resolve()
    if (not path.is_relative_to(base.resolve()) or original.is_symlink() or not path.is_file()
            or path.stat().st_size > 16*1024*1024 or digest(path)!=row.get("sha256")):
        raise ValueError("bound admission/control file differs")
    return path


def expected_control_sources(worker,plan):
    groups = {"component":("api/cached_jacobian.py","api/public_map_builder.py","api/component_controls.py"),
        "API":("api/matched_driver.py","api/normalized_exact.py","api/test_driver.py","api/protocol.json"),
        "worker":("realcase_worker.py","case_runtime.py","test_worker.py","project_protocol.json")}
    result = {key:{Path(name).name:digest(worker/name) for name in names} for key,names in groups.items()}
    result["parent"] = parent_contract()
    if set(result)!=set(plan["required_controls"]):raise ValueError("registered control groups differ")
    return result


def verify_controls(base,receipt,worker,plan):
    if (receipt.get("status")!="REGISTERED_PILOT_CONTROLS_ACTUAL_PASS" or receipt.get("runtime")!=plan["runtime"]
            or set(receipt.get("groups",{}))!=set(plan["required_controls"])):
        raise ValueError("fresh exact target-runtime control receipt required")
    expected = expected_control_sources(worker,plan)
    for name,count in plan["required_controls"].items():
        row = receipt["groups"][name]
        paths = {key:bound_file(base,row[key]) for key in ("log","result","source_identities","actual_wait")}
        identities = bounded_json(paths["source_identities"])
        if any(identities.get(k)!=v for k,v in expected[name].items()):raise ValueError("control source prehash differs")
        wait = bounded_json(paths["actual_wait"])
        if (wait.get("actual_wait_returncode")!=0 or wait.get("wait_observed") is not True
                or wait.get("terminal_state_known") is not True or wait.get("owned_process_group_alive_after_cleanup") is not False
                or type(wait.get("actual_pid")) is not int or wait.get("observed_pgid")!=wait["actual_pid"]):
            raise ValueError("actual control wait/group proof differs")
        if name=="parent":
            result = bounded_json(paths["result"])
            totals = {k:result.get(k) for k in ("tests","failures","errors","skipped")}
            ids = result.get("test_ids",[])
        else:
            root = ET.parse(paths["result"]).getroot()
            suites = [root] if root.tag=="testsuite" else list(root.findall("testsuite"))
            totals = {k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
            ids = [(c.get("classname"),c.get("name")) for c in root.iter("testcase")]
        if (totals!={"tests":count,"failures":0,"errors":0,"skipped":0}
                or len(ids)!=count or len(set(map(str,ids)))!=count):
            raise ValueError("actual unique frozen control counts differ")
    return {"groups":dict(plan["required_controls"]),"all_actual_counts_verified":True}


def observe_sources(repo,worker,plan,admission,admission_base):
    worker = worker.resolve();repo = repo.resolve()
    actual = {}
    for name,expected in plan["worker_contract"].items():
        original = worker/name;path = original.resolve()
        if not path.is_relative_to(worker) or original.is_symlink() or digest(path)!=expected:
            raise ValueError("frozen worker dependency changed: "+name)
        actual[name] = expected
    inventory = worker/"STAGED_FILE_MANIFEST.json"
    if digest(inventory)!=plan["worker_inventory_sha256"]:raise ValueError("worker inventory changed")
    for name,row in bounded_json(inventory)["files"].items():
        original = worker/name;path = original.resolve()
        if (not path.is_relative_to(worker) or original.is_symlink() or path.stat().st_size!=row["bytes"]
                or digest(path)!=row["sha256"]):raise ValueError("worker source/evidence changed: "+name)
    frozen = bounded_json(worker/"FROZEN_PRODUCTION_SOURCE.json")
    if frozen["source_sha256"]!=plan["production_source_sha256"] or len(frozen["files"])!=161:
        raise ValueError("production9f manifest differs")
    for name,expected in frozen["files"].items():
        path = (repo/name).resolve()
        if not path.is_relative_to(repo) or digest(path)!=expected:raise ValueError("production9f source changed")
    for name,expected in plan["raw_input_sha256"].items():
        if digest(repo/name)!=expected:raise ValueError("raw development input changed")
    roles = {name:digest(bound_file(admission_base,row)) for name,row in admission["roles"].items()}
    if roles.get("criteria")!=plan["criteria_sha256"] or roles.get("worker_review")!=plan["worker_review_sha256"]:
        raise ValueError("fixed worker review/technical criteria changed")
    return {"worker_contract":actual,"parent_contract":parent_contract(),"role_bindings":roles,
        "production_source_sha256":frozen["source_sha256"],"production_files":161,
        "worker_inventory_sha256":digest(inventory),"raw_input_sha256":dict(plan["raw_input_sha256"])}


def validate_go(go,plan,sources,admission_sha):
    flags = ("execution_authorized","independent_worker_source_review_clear","external_supervisor_review_clear",
        "independent_auditor_review_clear","runtime_review_clear","mandatory_frozen_controls_passed")
    if (go.get("authorization")!=AUTHORIZATION or any(go.get(k) is not True for k in flags)
            or go.get("mode")!=plan["mode"] or go.get("allowed_slot_indices")!=[0,1]
            or any(type(i) is not int for i in go.get("allowed_slot_indices",[]))
            or go.get("full48_authorized") is not False or go.get("source_files")!=sources["worker_contract"]
            or go.get("production_source_sha256")!=plan["production_source_sha256"]
            or go.get("supervisor_source_sha256")!=sources["parent_contract"]["serial_pilot.py"]
            or go.get("parent_contract")!=sources["parent_contract"] or go.get("admission_sha256")!=admission_sha
            or go.get("review_bindings")!=sources["role_bindings"]
            or go.get("auditor_source_sha256")!=sources["role_bindings"].get("auditor")
            or go.get("audit_criteria_sha256")!=plan["criteria_sha256"]):
        raise ValueError("separate exact reviewed pilot root GO required; no full48")
    if set(sources["role_bindings"])!={"auditor","criteria","controls","worker_review","parent_review","auditor_review","runtime_review"}:
        raise ValueError("all future review/auditor/control roles required")
    if any(re.fullmatch(r"[0-9a-f]{64}",str(x)) is None for x in sources["role_bindings"].values()):
        raise ValueError("exact role identities required")


def runtime_gate(plan):
    versions = {name:importlib.metadata.version(name) for name in ("numpy","scipy")}
    if (platform.system()!=plan["runtime"]["platform"] or platform.python_version()!=plan["runtime"]["python"]
            or any(versions[name]!=plan["runtime"][name] for name in versions)
            or any(os.environ.get(k)!=v for k,v in plan["threads"].items())):
        raise ValueError("exact target Linux/package/thread runtime required")
    return {"python":sys.version,"platform":platform.platform(),"packages":versions,"threads":dict(plan["threads"]),
        "numerical_imports_in_parent":0}


def group_exists(pgid):
    if pgid is None:return None
    try:os.killpg(pgid,0)
    except ProcessLookupError:return False
    return True


@contextmanager
def deferred_interruptions():
    pending,previous = [],{}
    def record(signum,frame):pending.append(signal.Signals(signum).name)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,record)
        yield pending
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def cleanup_owned(process,pgid,caps):
    actions = []
    for sig,grace in ((signal.SIGTERM,caps["TERM_grace_seconds"]),(signal.SIGKILL,caps["KILL_grace_seconds"])):
        if process.poll() is None or group_exists(pgid):
            try:
                if pgid is None:process.send_signal(sig)
                else:os.killpg(pgid,sig)
                actions.append(signal.Signals(sig).name)
            except ProcessLookupError:pass
        deadline = time.perf_counter()+grace
        while True:
            try:code = process.wait(timeout=max(.001,min(caps["cleanup_poll_seconds"],deadline-time.perf_counter())))
            except subprocess.TimeoutExpired:
                if time.perf_counter()>=deadline:break
                continue
            if not group_exists(pgid):
                return {"cleanup_signals":actions,"actual_wait_returncode":code,
                    "wait_observed":True,"owned_process_group_alive_after_cleanup":False}
            if time.perf_counter()>=deadline:break
            time.sleep(caps["cleanup_poll_seconds"])
    code = process.wait(timeout=caps["KILL_grace_seconds"])
    return {"cleanup_signals":actions,"actual_wait_returncode":code,"wait_observed":True,
        "owned_process_group_alive_after_cleanup":group_exists(pgid)}


def tree_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def run_owned(command,slot_dir,started,caps,env,pending):
    process = pgid = log = None
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = {"argv":list(map(str,command)),"absolute_started":started,"actual_pid":None,"observed_pgid":None,
        "requested_new_session":True,"launch_state":"NOT_ATTEMPTED","wait_observed":False,
        "terminal_state_known":False,"closed_log":False,"unknown_work":True,"post_wait_IO_errors":[]}
    try:
        write_new(slot_dir/"launch_requested.json",row)
        if pending:raise InterruptedError("parent stop before Popen")
        log = (slot_dir/"child.log").open("xb")
        row["launch_state"] = "POPEN_CONSTRUCTION_STARTED_OUTCOME_UNKNOWN"
        process = subprocess.Popen(command,env=env,cwd=slot_dir,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row.update(actual_pid=process.pid,launch_state="HANDLE_ASSIGNED")
        try:row["observed_pgid"]=os.getpgid(process.pid)
        except ProcessLookupError:pass
        if row["observed_pgid"]==process.pid:pgid=process.pid
        write_new(slot_dir/"spawned.json",row)
        if pgid is None:raise ValueError("actual new-session group not observed")
        while True:
            if pending:raise InterruptedError("parent stop request")
            remaining = caps["complete_process_seconds"]-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError("absolute complete-process deadline")
            if (slot_dir/"child.log").stat().st_size>caps["per_child_log_bytes"]:
                raise OSError("fixed child log cap exceeded")
            if tree_size(slot_dir)>caps["per_child_output_bytes"]+caps["per_child_log_bytes"]:
                raise OSError("fixed child output cap exceeded")
            try:
                row["actual_wait_returncode"]=process.wait(timeout=min(caps["poll_seconds"],remaining))
                row["wait_observed"]=True
                break
            except subprocess.TimeoutExpired:pass
    except BaseException as exc:  # noqa: BLE001 -- preserve failure plus owned actual wait
        row.update(exception_type=type(exc).__name__,message=str(exc))
    finally:
        if process is not None:
            try:
                row.update(cleanup_owned(process,pgid,caps))
                row["terminal_state_known"]=(pgid==process.pid and row["owned_process_group_alive_after_cleanup"] is False)
            except BaseException as exc:  # noqa: BLE001 -- no stable endpoint from unknown cleanup
                row["cleanup_error"]={"exception_type":type(exc).__name__,"message":str(exc)}
        elif row["launch_state"]=="NOT_ATTEMPTED":
            row["terminal_state_known"]=True  # proved no Popen attempt, unlike ambiguous constructor outcome
        if log is not None:
            try:log.close();row["closed_log"]=True
            except BaseException as exc:row["post_wait_IO_errors"].append({"phase":"logclose","exception_type":type(exc).__name__,"message":str(exc)})
        row["external_wait_cleanup_logclose_seconds"]=time.perf_counter()-started
    try:
        after = resource.getrusage(resource.RUSAGE_CHILDREN);scale = 1024 if platform.system()=="Linux" else 1
        row.update(child_CPU_delta_user_seconds=after.ru_utime-before.ru_utime,
            child_CPU_delta_system_seconds=after.ru_stime-before.ru_stime,
            parent_cumulative_children_peak_RSS_before_bytes=int(before.ru_maxrss)*scale,
            parent_cumulative_children_peak_RSS_after_bytes=int(after.ru_maxrss)*scale)
    except BaseException as exc:  # noqa: BLE001 -- resource observation cannot erase the actual wait
        row["post_wait_IO_errors"].append({"phase":"resource_observation","exception_type":type(exc).__name__,"message":str(exc)})
    row.update(resource_scope="CPU deltas include reaped direct/nested children in this serial interval; cumulative RUSAGE_CHILDREN RSS peak is never perworker RSS.",
        deferred_parent_signals=list(pending),signal_scope="Nonraising recorder through handle assignment, cleanup, logclose and receipt; no OS blocked signal mask.")
    row["unknown_work"]=not(row.get("actual_wait_returncode")==0 and row["terminal_state_known"]
        and row["closed_log"] and not pending and "exception_type" not in row and "cleanup_error" not in row)
    try:write_new(slot_dir/"actual_wait.json",row);row["actual_wait_receipt_sha256"]=digest(slot_dir/"actual_wait.json")
    except BaseException as exc:  # noqa: BLE001 -- actual exit in memory must survive postwait persistence failure
        row["post_wait_IO_errors"].append({"phase":"actual_wait_receipt","exception_type":type(exc).__name__,"message":str(exc)})
    return row


def read_worker_result(slot_dir,slot,worker_contract):
    worker = slot_dir/"worker";result_path=worker/"result.json";payload_path=worker/"returned_payload.json"
    result,payload = bounded_json(result_path),bounded_json(payload_path)
    if ((worker/"failure.json").exists() or result.get("status")!="WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT"
            or result.get("source_files")!=worker_contract or result.get("slot")!=slot
            or result.get("method_policy_valid") is not True or result.get("API_return_gate_only",{}).get("eligible") is not True
            or result.get("complete_OS_process_eligibility_available") is not False
            or payload.get("arm")!=slot["arm"] or payload.get("policy_valid") is not True
            or payload.get("unknown_work") is not False or payload.get("preparation_work_unknown") is not False
            or payload.get("error") is not None or payload.get("selected") is None
            or result.get("method_status")!=payload.get("status")
            or payload.get("selected_model") is None):
        raise ValueError("worker/API policyinvalid, failed, unknown or source/slot mismatch")
    age = payload["selected"].get("elapsed_seconds")
    if type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<=7.5:
        raise ValueError("selected accepted endpoint completed late/unknown")
    binding = result.get("case_binding",{})
    if (set(binding)!={"case_index","raw24_m_sha256","initial49_m_sha256","truth49_m_sha256","all8_observations_sha256","survey_sha256"}
            or type(binding["case_index"]) is not int or binding["case_index"]!=0
            or any(re.fullmatch(r"[0-9a-f]{64}",str(v)) is None for k,v in binding.items() if k!="case_index")):
        raise ValueError("complete same-study case binding required")
    return {"case_binding":binding,"method_status":payload["status"],"unknown_work":False,
        "worker_result_sha256":digest(result_path),"returned_payload_sha256":digest(payload_path),
        "API_flag_is_not_OS_proof":True}


def run_slot(slot,output,plan,worker_contract,command_factory,source_check,validator,env):
    started=time.perf_counter()  # before per-slot setup/admission/request/Popen
    directory=output/("slot_%02d"%slot["run_index"]);row={"slot":slot,"status":"attempted","absolute_started":started,
        "unknown_work":True,"eligible":False,"launch_state":"NOT_ATTEMPTED","terminal_state_known":True}
    with deferred_interruptions() as pending:
        try:
            directory.mkdir()
            source=source_check();write_new(directory/"source_before.json",source)
            request={"slot_index":slot["run_index"],"case_index":slot["case_index"],"arm":slot["arm"],
                "absolute_started":started,"optimizer_absolute_deadline":started+7.5,
                "complete_return_absolute_deadline":started+8.,"parent_pid":os.getpid(),
                "clock":"time.perf_counter_same_host",
                "source_contract_sha256":hashlib.sha256(canonical(worker_contract)).hexdigest()}
            write_new(directory/"request.json",request)
            command=command_factory(directory,request)
            row.update(run_owned(command,directory,started,plan["caps"],env,pending))
            if (row.get("actual_wait_returncode")!=0 or row.get("terminal_state_known") is not True
                    or row.get("closed_log") is not True or row.get("unknown_work") or row["post_wait_IO_errors"]):
                raise ValueError("actual child exit/cleanup/log/persistence is ineligible")
            row.update(validator(directory,slot,worker_contract))
            after=source_check();write_new(directory/"source_after.json",after)
            if after!=source:raise ValueError("per-slot source/input/admission changed")
            if pending:raise InterruptedError("parent stop after wait")
            row["eligible_before_external_return_observation"]=True
        except BaseException as exc:  # noqa: BLE001 -- keep actual wait and unknown prefix, stop later admission
            row.update(slot_exception_type=type(exc).__name__,slot_exception_message=str(exc),eligible_before_external_return_observation=False)
        row["deferred_parent_signals"]=list(pending)
        if pending:row["eligible_before_external_return_observation"]=False
        try:write_new(directory/"slot_return.json",row)
        except BaseException as exc:  # noqa: BLE001 -- preserve actual wait in caller even if serialization fails
            row.update(slot_return_persistence_error={"exception_type":type(exc).__name__,"message":str(exc)},eligible_before_external_return_observation=False)
    return row


def stable_inventory(output,attempts,caps):
    if any(r.get("status")=="attempted" and r.get("terminal_state_known") is not True for r in attempts):
        write_new(output/"UNSTABLE_CHILD_PREFIX_ONLY.json",{"complete_inventory":False,"unknown_child_terminal":True,
            "child_written_files_hashed":False,"actual_wait_rows_retained_in_attempts":True})
        return False
    rows={};used=0;complete=True
    for path in sorted(output.rglob("*")):
        if not path.is_file():continue
        relative=path.relative_to(output).as_posix()
        if relative=="artifact_inventory.json":continue
        if len(rows)>=caps["inventory_file_count"]:complete=False;break
        size=path.stat().st_size
        included=(not path.is_symlink() and size<=caps["individual_file_bytes"]
            and used+size<=caps["packet_bytes"]-caps["failure_reserve_bytes"])
        rows[relative]={"bytes":size,"sha256":digest(path) if included else None,"bounded_hash_completed":included}
        if included:used+=size
        else:complete=False
    write_new(output/"artifact_inventory.json",{"files":rows,"complete_inventory":complete,
        "hashed_bytes":used,"inventory_excludes_itself":True,"known_children_terminal_before_hashing":True,
        "scope":"Closed logs and child files after actual waits/owned-group cleanup; no mutable tee. No parent external-terminal proof."})
    return complete


def run_pair(output,plan,go,admission,sources,command_factory,source_check,validator,env):
    # Outer deferral also covers the run_slot return->caller handoff and final
    # metadata/inventory, when no child may be abandoned or later slot admitted.
    with deferred_interruptions() as pending:
        return _run_pair(output,plan,go,admission,sources,command_factory,source_check,validator,env,pending)


def _run_pair(output,plan,go,admission,sources,command_factory,source_check,validator,env,pending):
    output.mkdir();attempts=[{"slot":s,"status":"unattempted","reason":"not_yet_admitted"} for s in plan["slots"]]
    write_new(output/"project_plan.json",plan);write_new(output/"supplied_root_go.json",go)
    write_new(output/"admission.json",admission);write_new(output/"source_before.json",sources)
    replace_own(output/"attempts.json",attempts)
    binding=None;failure=None
    for i,slot in enumerate(plan["slots"]):
        if pending:
            failure={"phase":"before_slot_admission","parent_stop_signals":list(pending)}
            break
        row=run_slot(slot,output,plan,plan["worker_contract"],command_factory,source_check,validator,env)
        returned=time.perf_counter()  # immediately after fully serialized slot return, before interpretation
        row["external_return_observed_perf_counter"]=returned
        row["external_complete_return_seconds"]=returned-row["absolute_started"]
        row["eligible"]=(row.get("eligible_before_external_return_observation") is True
            and row["external_complete_return_seconds"]<=plan["caps"]["complete_process_seconds"] and not pending)
        row["parent_signals_outside_slot"]=list(pending)
        if row["eligible"]:
            if binding is None:binding=row["case_binding"]
            elif row["case_binding"]!=binding:
                row.update(eligible=False,binding_mismatch=True)
        attempts[i]=row
        try:replace_own(output/"attempts.json",attempts)
        except BaseException as exc:  # noqa: BLE001 -- do not replace actual exit with persistence exception
            failure={"exception_type":type(exc).__name__,"message":str(exc),"phase":"attempts_persistence"};break
        if not row["eligible"]:break
    for row in attempts:
        if row["status"]=="unattempted":row["reason"]="stopped_after_ineligible_failed_unknown_or_late_prior_slot"
    try:
        replace_own(output/"attempts.json",attempts)
        after=source_check();write_new(output/"source_after.json",after)
        if after!=sources:raise ValueError("final source binding changed")
        report={"status":"PILOT_TWO_RETURNS_PENDING_INDEPENDENT_AUDIT_AND_PARENT_EXTERNAL_WAIT" if all(r.get("eligible") for r in attempts) and failure is None else "PILOT_FAILED_OR_INCOMPLETE_NO_RETRY",
            "attempts":attempts,"unknown_work":any(r.get("unknown_work",True) for r in attempts),
            "failure":failure,"deferred_parent_signals":list(pending),"full48_authorized":False,"parent_terminal_externally_observed":False,
            "scientific_goals_complete":False,"independent_audit_required":True}
        write_new(output/"pilot_result.json",report)
        inventory_ok=stable_inventory(output,attempts,plan["caps"])
        return (0 if all(r.get("eligible") for r in attempts) and failure is None and inventory_ok and not pending else 2),report
    except BaseException as exc:  # noqa: BLE001 -- every actual wait row remains in returned failure prefix
        report={"status":"PILOT_POSTWAIT_PERSISTENCE_OR_INVENTORY_FAILURE","attempts":attempts,
            "exception_type":type(exc).__name__,"message":str(exc),"prior_failure":failure,
            "parent_terminal_externally_observed":False,"scientific_goals_complete":False}
        try:write_new(output/"parent_failure_prefix.json",report)
        except BaseException:
            print(json.dumps({"status":"PARENT_FAILURE_PREFIX_IO_FAILED","actual_wait_prefixes":[{k:r.get(k) for k in ("actual_pid","observed_pgid","actual_wait_returncode","wait_observed","terminal_state_known","closed_log","launch_state")} for r in attempts]}))
        return 2,report


def main(argv=None):
    entered=time.perf_counter();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    for name in ("repo","worker-dir","root-go","admission","output"):parser.add_argument("--"+name,type=Path)
    args=parser.parse_args(argv);plan=bounded_json(HERE/"PILOT_PLAN.json")
    if not args.execute:
        print(json.dumps({"status":"HELD_SOURCE_ONLY_SERIAL_CASE0_PARENT","source_files":parent_contract(),
            "project_execution_authorized":False,"numerical_imports":0,"PDE_calls":0,"full48_authorized":False}))
        return 0
    if any(getattr(args,k) is None for k in ("repo","worker_dir","root_go","admission","output")):
        parser.error("explicit repo/worker-dir/root-go/admission/new output required")
    try:
        worker=args.worker_dir.resolve();repo=args.repo.resolve();output=args.output.resolve()
        if output.is_relative_to(worker) or output.is_relative_to(repo) or output.is_relative_to(HERE):
            raise ValueError("output must be outside all source roots")
        admission=bounded_json(args.admission);go=bounded_json(args.root_go)
        sources=observe_sources(repo,worker,plan,admission,args.admission.resolve().parent)
        validate_go(go,plan,sources,digest(args.admission))
        runtime=runtime_gate(plan)
        controls=verify_controls(args.admission.resolve().parent,bounded_json(bound_file(args.admission.resolve().parent,admission["roles"]["controls"])),worker,plan)
        admission={**admission,"observed_sources":sources,"observed_runtime":runtime,"registered_controls":controls,
            "shared_parent_admission_seconds":time.perf_counter()-entered,
            "shared_cost_scope":"Measured separately; includes CLI/source/GO/runtime/control checks after stdlib imports. Interpreter/parent-import cold startup and final parent OS exit require future external observer; no fulltotalcost claim here."}
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",**plan["threads"])
        for key in ("GITHUB_TOKEN","GH_TOKEN","ACTIONS_RUNTIME_TOKEN","ACTIONS_ID_TOKEN_REQUEST_TOKEN","ACTIONS_ID_TOKEN_REQUEST_URL"):env.pop(key,None)
        def command(directory,request):
            return [sys.executable,"-B","-I",str(worker/"realcase_worker.py"),"--execute","--repo",str(repo),
                "--root-go",str(output/"supplied_root_go.json"),"--request",str(directory/"request.json"),"--output",str(directory/"worker")]
        return run_pair(output,plan,go,admission,sources,command,
            lambda:observe_sources(repo,worker,plan,admission,args.admission.resolve().parent),read_worker_result,env)[0]
    except BaseException as exc:  # noqa: BLE001 -- no real worker exists before successful admission
        print(json.dumps({"status":"PILOT_ADMISSION_OR_PARENT_FAILURE","exception_type":type(exc).__name__,
            "message":str(exc),"execution_retry_available":False,"scientific_goals_complete":False}))
        return 2


if __name__=="__main__":
    raise SystemExit(main())
