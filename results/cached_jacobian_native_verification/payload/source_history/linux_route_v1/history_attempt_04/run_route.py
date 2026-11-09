"""Held first-push Linux route:58 controls, then one conditionally authorized child.

Standard library only. No install, model, numerical import, retry or daemon.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPOSITORY="uskutsav-cpu/wavecert-fwi"
BRANCH="research-cached-j-native-01"
BASELINE="2918a47c68ac8bee3d33e532c21a96d335d1718a"
RUNTIME={
    "verify_native_fields.py":"c660a025badbebd5e9d9a4d6ddbfc684c95193c6111f2de349f25112976d1d6b",
    "protocol.json":"3b608f942f3d08f40705e8203375fc7e24763592bd27d7cc70bac12b351391e6",
    "FROZEN_PRODUCTION_SOURCE.json":"e7ba5deef7a651f42d02b9f8e0bcbee9624291b8217ab378bdde7586c57cd4de",
    "cached_jacobian.py":"f613370617d7e0bd7815cf441dd8ba5f41816ad4791275e86f33de228657c79b",
    "public_map_builder.py":"622a624f54b17d9d882da9d88af547c516b26ee42c87d80ec68d5fe99a687ec1",
    "test_lifecycle.py":"2526a76b539af952a9a9707f6723493ad571723012ed48c8e3738b8432a80414",
    "test_manufactured.py":"c7d7d9d9e152b4b0c037b5cd2efc411356d2c1206b5ce3824ed8a351acda3104"}
WORKER_CONTRACT={name:RUNTIME[name] for name in tuple(RUNTIME)[:5]}
ARTIFACT_CAP=32*1024*1024
LOG_CAP=4*1024*1024


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],text=True,timeout=10).strip()


def runtime_gate():
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pytest")}
    if (platform.system()!="Linux" or platform.python_version()!="3.11.18"
            or versions!={"numpy":"2.4.6","scipy":"1.17.1","pytest":"9.1.1"}):
        raise ValueError("exact ephemeral Linux CPU verification runtime differs")
    return {"python":sys.version,"platform":platform.platform(),"packages":versions,"numerical_imports_in_controller":0}


def event_gate(event,env,head,parents,baseline):
    repo=event.get("repository",{})
    if (env.get("GITHUB_EVENT_NAME")!="push" or env.get("GITHUB_REPOSITORY")!=REPOSITORY
            or env.get("GITHUB_REF")!="refs/heads/"+BRANCH or env.get("GITHUB_RUN_ATTEMPT")!="1"
            or event.get("ref")!="refs/heads/"+BRANCH or event.get("created") is not True
            or event.get("forced") is not False or event.get("deleted") is not False
            or event.get("before")!="0"*40 or repo.get("full_name")!=REPOSITORY
            or repo.get("private") is not False or repo.get("visibility","public")!="public"
            or event.get("after")!=head or event.get("head_commit",{}).get("id")!=head
            or env.get("GITHUB_SHA")!=head or parents!=[head,BASELINE] or baseline!=BASELINE
            or env.get("GITHUB_WORKFLOW_SHA")!=head
            or env.get("GITHUB_WORKFLOW_REF")!=REPOSITORY+"/.github/workflows/cached-j-native.yml@refs/heads/"+BRANCH):
        raise ValueError("actual public nonforced first push/attempt1/single-parent291 context differs")


def source_gate(experiment,baseline):
    manifest_path=HERE/"route_manifest.json"
    manifest=json.loads(manifest_path.read_bytes())
    for relative,row in manifest["files"].items():
        original=experiment/relative;path=original.resolve()
        if not path.is_relative_to(experiment) or original.is_symlink() or digest(path)!=row["sha256"] or path.stat().st_size!=row["bytes"]:
            raise ValueError("reviewed route/workflow file changed: "+relative)
    for name,expected in RUNTIME.items():
        if digest(HERE/"runtime"/name)!=expected:raise ValueError("reviewed exact runtime copy changed")
    frozen=json.loads((HERE/"runtime/FROZEN_PRODUCTION_SOURCE.json").read_bytes())
    for relative,expected in frozen["files"].items():
        if digest(baseline/relative)!=expected:raise ValueError("baseline production9f file changed")
    # The complete production manifest is independently checked by c660 again.
    result={"route_manifest_sha256":digest(manifest_path),"route_files":manifest["files"],
        "baseline_source_sha256":frozen["source_sha256"],"baseline_files":frozen["files"],
        "experiment_actual_head":git(experiment,"rev-parse","HEAD"),"baseline_actual_head":git(baseline,"rev-parse","HEAD"),
        "experiment_worktree_status":git(experiment,"status","--porcelain","--untracked-files=all"),
        "baseline_worktree_status":git(baseline,"status","--porcelain","--untracked-files=all")}
    if (result["baseline_actual_head"]!=BASELINE or result["experiment_worktree_status"] or result["baseline_worktree_status"]):
        raise ValueError("separate exact baseline and experiment checkouts must remain clean")
    return result


def registration_gate(registration,manifest_sha):
    if (registration.get("repository")!=REPOSITORY or registration.get("branch")!=BRANCH
            or registration.get("single_parent")!=BASELINE or registration.get("route_manifest_sha256")!=manifest_sha
            or type(registration.get("native_execution_registered")) is not bool):
        raise ValueError("route registration differs from reviewed branch/source context")
    enabled=registration["native_execution_registered"]
    go=registration.get("root_go")
    if not enabled:
        if go is not None:raise ValueError("false native registration must have null root GO")
        return None
    registration_sha=hashlib.sha256(canonical({k:v for k,v in registration.items() if k!="root_go"})).hexdigest()
    if (not isinstance(go,dict) or go.get("authorization")!="GO_CACHED_J_NATIVE_FIELD_VERIFICATION"
            or go.get("execution_authorized") is not True or go.get("independent_source_review_clear") is not True
            or go.get("criteria_frozen_before_execution") is not True or go.get("source_files")!=WORKER_CONTRACT
            or go.get("route_manifest_sha256")!=manifest_sha or go.get("route_registration_sha256")!=registration_sha):
        raise ValueError("conditional native run requires supplied exact root GO after local reviews/freeze")
    return go


def group_exists(pgid):
    if pgid is None:return None
    try:os.killpg(pgid,0)
    except ProcessLookupError:return False
    return True


@contextmanager
def deferred_interruptions():
    """Record parent stop requests until ownership/cleanup evidence is durable.

    This changes handlers, never the signal mask inherited by an exec child.
    The recorder remains nonraising through repeated signals during cleanup.
    """
    pending=[];previous={}
    def record(signum,frame):
        pending.append(signal.Signals(signum).name)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.signal(sig,record)
        yield pending
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def reap(process,pgid):
    """Only an owned new session; every direct child gets an actual wait."""
    actions=[]
    for sig,grace in ((signal.SIGTERM,3.),(signal.SIGKILL,3.)):
        if process.poll() is None or group_exists(pgid):
            try:
                if pgid is None:process.send_signal(sig)
                else:os.killpg(pgid,sig)
                actions.append(signal.Signals(sig).name)
            except ProcessLookupError:pass
        try:process.wait(timeout=grace)
        except subprocess.TimeoutExpired:continue
        if not group_exists(pgid):break
        time.sleep(.05)
    return {"signals":actions,"actual_wait_returncode":process.wait(timeout=3.),
        "owned_process_group_alive_after_cleanup":group_exists(pgid)}


def run_child(label,command,*,cwd,raw,env,timeout):
    started=time.perf_counter();process=None;pgid=None;interrupted=None
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    row={"label":label,"argv":list(map(str,command)),"cwd":str(cwd),"timeout_seconds":timeout,
        "requested_new_session":True,"actual_pid":None,"observed_pgid":None,"wait_observed":False,
        "launch_state":"NOT_ATTEMPTED","terminal_state_known":False}
    write(raw/(label+"_requested.json"),row)
    with deferred_interruptions() as pending:
        with (raw/(label+".log")).open("xb") as log:
            try:
                row["launch_state"]="POPEN_CONSTRUCTION_STARTED_OUTCOME_UNKNOWN"
                process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(actual_pid=process.pid,launch_state="HANDLE_ASSIGNED")
                try:row["observed_pgid"]=os.getpgid(process.pid)
                except ProcessLookupError:pass
                if row["observed_pgid"]==process.pid:pgid=process.pid
                write(raw/(label+"_spawned.json"),row)
                if pgid is None:raise ValueError("owned new-session group identity not known")
                while True:
                    if pending:raise InterruptedError("parent stop request: "+",".join(pending))
                    remaining=timeout-(time.perf_counter()-started)
                    if remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
                    try:
                        row["actual_wait_returncode"]=process.wait(timeout=min(.5,remaining))
                        row["wait_observed"]=True
                        break
                    except subprocess.TimeoutExpired:
                        if (raw/(label+".log")).stat().st_size>LOG_CAP:raise ValueError("fixed child log cap exceeded")
            except BaseException as exc:  # noqa: BLE001 -- always reap owned child before evidence inventory
                interrupted=exc;row["exception_type"]=type(exc).__name__;row["message"]=str(exc)
            finally:
                if process is not None:
                    try:
                        cleanup=reap(process,pgid)
                        row.update(cleanup,wait_observed=True)
                        row["terminal_state_known"]=cleanup["owned_process_group_alive_after_cleanup"] is False
                    except BaseException as exc:  # noqa: BLE001 -- retain unknown cleanup without stable-end claims
                        row["cleanup_exception_type"]=type(exc).__name__;row["cleanup_message"]=str(exc)
                        if interrupted is None:interrupted=exc
                # Even an OSError from Popen has no returned handle here. Do not
                # infer a confirmed childless failure from actual_pid=None.
                row["external_spawn_through_wait_wall_seconds"]=time.perf_counter()-started
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        scale=1024 if platform.system()=="Linux" else 1
        row.update(child_CPU_delta_user_seconds=after.ru_utime-before.ru_utime,
            child_CPU_delta_system_seconds=after.ru_stime-before.ru_stime,
            parent_cumulative_children_peak_RSS_before_bytes=int(before.ru_maxrss)*scale,
            parent_cumulative_children_peak_RSS_after_bytes=int(after.ru_maxrss)*scale,
            deferred_parent_signals=pending,
            signal_scope="Nonraising handlers through Popen handle handoff, owned cleanup, closed log and actual-wait receipt; no blocked child signal mask.",
            RSS_scope="Cumulative RUSAGE_CHILDREN peak includes earlier control/nested children; never assigned solely to native. Native self peak is in its own report, before final serialization.")
        write(raw/(label+"_actual_wait.json"),row)
        if pending and interrupted is None:interrupted=InterruptedError("parent stop request: "+",".join(pending))
    if interrupted is not None:raise interrupted
    if row.get("owned_process_group_alive_after_cleanup") is not False:raise ValueError("owned child group cleanup not known complete")
    return row


def check_controls(raw):
    text=(raw/"lifecycle14.log").read_text()
    value=json.loads(text.strip().splitlines()[-1])
    if ((value.get("tests"),value.get("failures"),value.get("errors"))!=(14,0,0)
            or value.get("source_files")!=WORKER_CONTRACT or value.get("status")!="MANUFACTURED_LIFECYCLE_PASS"
            or value.get("control_source_sha256")!=RUNTIME["test_lifecycle.py"]
            or any(value.get(key)!=0 for key in ("numerical_imports","native_owners_constructed","PDE_calls"))):
        raise ValueError("actual isolated14 lifecycle count/source differs")
    if (len(re.findall(r"^test_.* \.\.\. ok$",text,re.MULTILINE))!=14
            or "... skipped" in text or "OK (skipped=" in text):raise ValueError("lifecycle skips/incomplete cases rejected")
    root=ET.parse(raw/"component44.xml").getroot()
    suites=[root] if root.tag=="testsuite" else list(root.findall("testsuite"))
    totals={key:sum(int(s.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
    if totals!={"tests":44,"failures":0,"errors":0,"skipped":0} or len(root.findall(".//testcase"))!=44:
        raise ValueError("actual44 component tests must all pass without errors/skips")
    return {"lifecycle14":value,"component44":totals,"total_mandatory_tests":58}


def artifact_packet(raw,artifact):
    """Only after all known children are reaped; immutable logs, no tee."""
    artifact.mkdir()
    for request in raw.glob("*_requested.json"):
        wait=raw/(request.name.removesuffix("_requested.json")+"_actual_wait.json")
        row=json.loads(wait.read_bytes()) if wait.exists() else None
        if (row is None or row.get("terminal_state_known") is not True
                or row.get("launch_state")!="HANDLE_ASSIGNED" or row.get("actual_pid") is None
                or not row.get("wait_observed") or row.get("owned_process_group_alive_after_cleanup") is not False):
            # Own immutable launch/failure metadata can still be retained, but
            # child-written logs/events cannot receive a stable-end inventory.
            for own in raw.glob("*.json"):
                if own.name.endswith(("_requested.json","_spawned.json","_actual_wait.json")) or own.name=="route_failure.json":
                    shutil.copyfile(own,artifact/own.name)
            write(artifact/"UNSTABLE_CHILD_PREFIX_ONLY.json",{"complete_inventory":False,"reason":"actual child/group terminal state not known; no stable child-file hashes claimed"})
            raise ValueError("known terminal state is required before stable artifact inventory")
    paths=[p for p in raw.rglob("*") if p.is_file()]
    paths.sort(key=lambda p:(p.suffix==".npz",p.relative_to(raw).as_posix()))
    inventory={};used=0
    for path in paths:
        if path.is_symlink():raise ValueError("artifact source cannot be a symlink")
        relative=path.relative_to(raw).as_posix();size=path.stat().st_size
        row={"sha256":digest(path),"bytes":size,"included":used+size<=ARTIFACT_CAP-1024*1024}
        inventory[relative]=row
        if row["included"]:
            target=artifact/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);used+=size
    write(artifact/"artifact_inventory.json",{"files":inventory,"copied_bytes":used,
        "artifact_cap_bytes":ARTIFACT_CAP,"inventory_excludes_itself":True,
        "any_omitted":any(not row["included"] for row in inventory.values()),
        "scope":"Hashes after known direct-child waits and owned-group cleanup. Over-cap payloads remain on runner, explicitly omitted; prefix metadata/logs precede arrays. No truncated mutable log."})
    if any(not row["included"] for row in inventory.values()):raise ValueError("bounded artifact omitted over-cap payload; no full evidence acceptance")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();experiment=HERE.parents[1].resolve();baseline=args.baseline.resolve();output=args.output.resolve()
    if output.is_relative_to(experiment) or output.is_relative_to(baseline):raise ValueError("output must be separate from both source checkouts")
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir();raw=output/"raw";raw.mkdir()
    status=2
    try:
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
        env={key:os.environ.get(key) for key in ("GITHUB_EVENT_NAME","GITHUB_REPOSITORY","GITHUB_REF","GITHUB_RUN_ATTEMPT","GITHUB_SHA","GITHUB_WORKFLOW_SHA","GITHUB_WORKFLOW_REF","GITHUB_RUN_ID")}
        write(raw/"actual_github_event.json",event);write(raw/"actual_github_context.json",env)
        event_gate(event,env,git(experiment,"rev-parse","HEAD"),git(experiment,"rev-list","--parents","-n","1","HEAD").split(),git(baseline,"rev-parse","HEAD"))
        write(raw/"actual_runtime.json",runtime_gate())
        before=source_gate(experiment,baseline);write(raw/"source_before.json",before)
        registration=json.loads((HERE/"registration.json").read_bytes());write(raw/"actual_registration.json",registration)
        go=registration_gate(registration,before["route_manifest_sha256"])
        temporary=raw/"tmp";temporary.mkdir()
        child_env=dict(os.environ,TMPDIR=str(temporary),PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for key in ("GITHUB_TOKEN","GH_TOKEN","ACTIONS_RUNTIME_TOKEN","ACTIONS_ID_TOKEN_REQUEST_TOKEN","ACTIONS_ID_TOKEN_REQUEST_URL"):child_env.pop(key,None)
        runtime=HERE/"runtime"
        first=run_child("lifecycle14",[sys.executable,"-B","-I",str(runtime/"test_lifecycle.py")],cwd=runtime,raw=raw,env=child_env,timeout=120)
        if first["actual_wait_returncode"]!=0:raise ValueError("isolated lifecycle control child failed")
        second=run_child("component44",[sys.executable,"-B","-I","-m","pytest","-q","-p","no:cacheprovider",str(runtime/"test_manufactured.py"),"--basetemp",str(temporary/"pytest"),"--junitxml",str(raw/"component44.xml")],cwd=runtime,raw=raw,env=child_env,timeout=120)
        if second["actual_wait_returncode"]!=0:raise ValueError("component control child failed")
        controls=check_controls(raw);write(raw/"mandatory58_controls.json",controls)
        if source_gate(experiment,baseline)!=before:raise ValueError("sources changed during controls")
        if go is None:
            report={"status":"CONTROLS58_PASS_NATIVE_NOT_REGISTERED","native_child_attempted":False}
        else:
            write(raw/"supplied_root_go.json",go)
            native=run_child("native",[sys.executable,"-B","-I",str(runtime/"verify_native_fields.py"),"--execute","--repo",str(baseline),"--root-go",str(raw/"supplied_root_go.json"),"--output",str(raw/"native")],cwd=runtime,raw=raw,env=child_env,timeout=240)
            if native["actual_wait_returncode"]!=0:raise ValueError("actual native child exit was nonzero or signalled")
            result=json.loads((raw/"native/result.json").read_bytes())
            if ((raw/"native/failure.json").exists() or result.get("source_files")!=WORKER_CONTRACT
                    or result.get("status")!="NATIVE_FIELD_INTEGRATION_CHECKS_PASS_PENDING_EXTERNAL_WAIT"):
                raise ValueError("native self-report absent/inconsistent despite actual exit0")
            report={"status":"NATIVE_CHILD_ACTUAL_EXIT0_SELF_REPORT_PRESENT_AUDIT_PENDING","native_child_attempted":True,"actual_native_wait":native}
        after=source_gate(experiment,baseline);write(raw/"source_after.json",after)
        if after!=before:raise ValueError("final source bytes changed")
        write(raw/"route_result.json",{**report,"mandatory_controls":58,"scientific_goals_complete":False,"independent_numerical_audit_required":True})
        status=0
    except BaseException as exc:  # noqa: BLE001 -- preserve source/event/actual-wait prefixes without retry
        write(raw/"route_failure.json",{"status":"FAILED_OR_INTERRUPTED_NO_RETRY","exception_type":type(exc).__name__,"message":str(exc),"scientific_goals_complete":False})
    finally:
        try:artifact_packet(raw,output/"artifact")
        except BaseException as exc:  # noqa: BLE001 -- cannot award complete credit after packaging failure
            print(json.dumps({"status":"ARTIFACT_PACKET_FAILED","exception_type":type(exc).__name__,"message":str(exc)}));status=2
    return status


if __name__=="__main__":
    def interrupted(signum,frame):
        raise InterruptedError("parent received "+signal.Signals(signum).name)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    raise SystemExit(main())
