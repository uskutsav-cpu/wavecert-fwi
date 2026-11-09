"""Held checkpoint23 environment-only caller. Default creates no files or children."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,platform,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location("checkpoint23_environment_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
S=module("process_supervision");E=module("environment_steps")
REPOSITORY="uskutsav-cpu/wavecert-fwi"
BRANCH="research-checkpoint23-environment-01"
TASK=Path("/opt/wavecert/checkpoint23_environment_01")
PREFIX=Path("/opt/wavecert/checkpoint23/env_01")
V2_BUNDLE="c8f6cd4a69301df8a04f014746a960eb925cb6acb5346fdfc5a7d5e87901ff3c"
CORE=("prepare_environment.py","environment_steps.py","process_supervision.py","namespace_gate.py","import_identity_probe.py","test_environment_route.py","test_import_identity_probe.py","run_probe_controls.py","runtime_top_constraints.in","build_requirements.in","SOURCE_BUILD_ALLOWLIST.json","frozen_v2/worker/protocol.json","frozen_v2/worker/RUNTIME_SOURCE_PINS.json")
EXPECTED_FILES={"research/checkpoint23_environment/"+n for n in CORE}|{".github/workflows/checkpoint23-environment-01.yml",".github/workflows/ci.yml"}
THREADS={k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")}
NATIVE_ENV={"DEVITO_LANGUAGE":"C","DEVITO_MPI":"0","DEVITO_AUTOTUNING":"off","DEVITO_AUTOPADDING":"0","DEVITO_SAFE_MATH":"1","DEVITO_JIT_BACKDOOR":"0","DEVITO_FIRST_TOUCH":"0","MPLBACKEND":"Agg"}

def controlled(tmp):
    env={k:v for k,v in os.environ.items() if not k.startswith(("PIP_","DEVITO_")) and k not in ("GITHUB_TOKEN","GH_TOKEN","ACTIONS_RUNTIME_TOKEN","ACTIONS_ID_TOKEN_REQUEST_TOKEN","PYTHONPATH","PYTHONHOME","LD_PRELOAD")}
    env.update(THREADS);env.update(NATIVE_ENV)
    env.update(TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",PYTHONHASHSEED="0",PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK="1",PIP_NO_CACHE_DIR="1",PIP_RETRIES="0")
    return env

def source_gate(repo,event,context,registration):
    manifest=E.read(HERE/"ROUTE_MANIFEST.json")
    if manifest.get("schema")!="checkpoint23_environment_route_bundle_v1" or set(manifest.get("files",{}))!=EXPECTED_FILES:raise ValueError("complete exact route/workflow source denominator required")
    for name,row in manifest["files"].items():
        p=repo/name
        if p.is_symlink() or not p.resolve().is_relative_to(repo) or p.stat().st_size!=row["bytes"] or S.digest(p)!=row["sha256"]:raise ValueError("route/workflow source differs: "+name)
    if (context.get("GITHUB_REPOSITORY")!=REPOSITORY or context.get("GITHUB_REF")!="refs/heads/"+BRANCH
            or context.get("GITHUB_EVENT_NAME")!="push" or context.get("GITHUB_RUN_ATTEMPT")!="1"
            or context.get("GITHUB_WORKFLOW_SHA")!=context.get("GITHUB_SHA")
            or context.get("GITHUB_WORKFLOW_REF")!=REPOSITORY+"/.github/workflows/checkpoint23-environment-01.yml@refs/heads/"+BRANCH
            or event.get("forced") is not False or event.get("deleted") is not False
            or event.get("repository",{}).get("private") is not False or event.get("after")!=context.get("GITHUB_SHA")):raise ValueError("exact public branch push/attempt1 context required")
    if (registration.get("schema")!="checkpoint23_environment_preparation_registration_v1"
            or registration.get("route_manifest_sha256")!=S.digest(HERE/"ROUTE_MANIFEST.json")
            or registration.get("v2_runtime_bundle_sha256")!=V2_BUNDLE
            or registration.get("native_execution_authorized") is not False
            or type(registration.get("environment_preparation_registered")) is not bool):raise ValueError("separate environment-only registration required")
    go=registration.get("root_go")
    if not registration["environment_preparation_registered"]:
        if go is not None:raise ValueError("held registration must have null GO")
        return None,manifest
    expected=hashlib.sha256(json.dumps({k:v for k,v in registration.items() if k!="root_go"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if (not isinstance(go,dict) or go.get("authorization")!="GO_CHECKPOINT23_ENVIRONMENT_PREPARATION_V1"
            or go.get("execution_authorized") is not True or go.get("native_execution_authorized") is not False
            or go.get("import_only_identity_probe_authorized") is not True
            or go.get("route_manifest_sha256")!=S.digest(HERE/"ROUTE_MANIFEST.json")
            or go.get("registration_without_go_sha256")!=expected or go.get("source_files")!=manifest["files"]
            or go.get("root_environment_preparation_source_review_clear") is not True
            or not registration.get("source_bundle_commit") or type(registration.get("source_control_run_id")) is not int):raise ValueError("exact reviewed preparation-only GO required")
    scope=go.get("import_identity_probe")
    probe_sources={"import_identity_probe.py":S.digest(HERE/"import_identity_probe.py"),"protocol.json":S.digest(HERE/"frozen_v2/worker/protocol.json"),"RUNTIME_SOURCE_PINS.json":S.digest(HERE/"frozen_v2/worker/RUNTIME_SOURCE_PINS.json")}
    if not isinstance(scope,dict) or scope.get("source_files")!=probe_sources or scope.get("prefix")!=str(PREFIX):raise ValueError("exact nested import-only probe projection required before package work")
    return go,manifest

def activation_gate(head,parents,changed,registration):
    if parents!=[head,registration["source_bundle_commit"]] or changed!=["research/checkpoint23_environment/RUN_REGISTRATION_01.json"]:raise ValueError("registration-only direct child of reviewed source/control commit required")

class Route:
    def __init__(self,metadata):
        self.here=HERE;self.root=TASK;self.prefix=PREFIX;self.metadata=metadata
        self.raw=metadata.parent/"checkpoint23_environment_raw";self.raw.mkdir()
        self.tmp=self.raw/"tmp";self.tmp.mkdir();self.env=controlled(self.tmp)
        self.fetch=E.Fetch();self.built={};self.sequence=0
    def observe(self,*_):
        if E.size(self.root)+E.size(self.prefix)+E.size(self.raw)>E.CAP["workspace"] or shutil.disk_usage("/opt").free<E.CAP["free"]:raise ValueError("observed workspace/free-space cap exceeded")
    def child(self,label,argv,observer=None,allowed=(0,)):
        self.sequence+=1;label=f"{self.sequence:02d}_{label}"
        row=S.run_child(label,list(map(str,argv)),cwd=self.raw,raw=self.raw,env=self.env,timeout=600,observer=observer or self.observe)
        if row["actual_wait_returncode"] not in allowed:raise ValueError("actual child exit failed: "+label)
        return row,self.raw/(label+".log")
    def namespace(self,python,output,extra):
        return ["/usr/bin/sudo","-n","/usr/bin/unshare","--net","--","/usr/bin/setpriv",f"--reuid={os.getuid()}",f"--regid={os.getgid()}","--clear-groups","--no-new-privs","--bounding-set=-all","--inh-caps=-all","--ambient-caps=-all","--",str(python),"-B","-I",str(HERE/"namespace_gate.py"),"--execute","--parent-netns",os.readlink("/proc/self/ns/net"),"--uid",str(os.getuid()),"--gid",str(os.getgid()),"--output",str(output),*map(str,extra)]
    def packet(self):
        known=True
        for requested in self.raw.glob("*_requested.json"):
            wait=requested.with_name(requested.name.replace("_requested.json","_actual_wait.json"))
            row=E.read(wait) if wait.exists() else {}
            if row.get("terminal_state_known") is not True or row.get("owned_process_group_alive_after_cleanup") is not False:known=False
        if known:E.copy_packet(self.raw,self.metadata,E.CAP["metadata"])
        else:
            self.metadata.mkdir(exist_ok=True)
            for p in self.raw.glob("*.json"):
                if p.name.endswith(("_requested.json","_spawned.json","_actual_wait.json")) or p.name=="failure.json":shutil.copyfile(p,self.metadata/p.name)
            E.write(self.metadata/"UNSTABLE_PREFIX_ONLY.json",{"complete_inventory":False,"unknown_work":True,"reason":"actual child/group terminal state unknown; no stable child log/payload hash claim"})
        return known

def prepare(go,metadata):
    if platform.system()!="Linux" or platform.machine()!="x86_64" or platform.python_version()!="3.11.18" or os.getuid()==0:raise ValueError("exact nonroot ephemeral Linux/Python3.11.18 required")
    if TASK.exists() or PREFIX.parent.exists() or shutil.disk_usage("/opt").free<E.CAP["free"]:raise ValueError("fresh stable task roots and headroom required")
    route=Route(metadata);started=time.perf_counter();status=2
    try:
        E.write(route.raw/"root_preparation_go.json",go)
        E.write(route.raw/"controller_runtime.json",{"python":sys.version,"executable":sys.executable,"realpath":str(Path(sys.executable).resolve()),"binary_sha256":S.digest(Path(sys.executable).resolve()),"platform":platform.platform(),"parent_controlled_environment":{k:route.env[k] for k in THREADS|NATIVE_ENV},"native_use_review_clear":False,"PID":os.getpid(),"PPID":os.getppid(),"PGID":os.getpgrp(),"SID":os.getsid(0)})
        # One Linux stdlib dummy gate before any package acquisition or build.
        proof=route.raw/"namespace_dummy.json";sent=[]
        def signal_probe(process,pgid):
            route.observe()
            if proof.with_suffix(".ready").exists() and not sent:os.killpg(pgid,signal.SIGTERM);sent.append(True)
        route.child("namespace_dummy",route.namespace(sys.executable,proof,["--dummy-signal"]),signal_probe,allowed=(0,-signal.SIGTERM,128+signal.SIGTERM))
        gate=E.read(proof)
        if gate.get("status")!="ACTUAL_LINUX_DUMMY_NAMESPACE_SIGNAL_PASS" or not sent or not gate.get("actual_signal_numbers"):raise ValueError("actual namespace/signal evidence absent")
        route.child("create_task_roots",["/usr/bin/sudo","-n","/usr/bin/install","-d","-m","0700","-o",os.getuid(),"-g",os.getgid(),TASK,PREFIX.parent])
        result=E.pipeline(route,go)
        result.update(controller_pre_result_interval_seconds=time.perf_counter()-started,transport_scope="Explicit downloader body bytes recorded; pip resolver transfer bytes are unknown.",caps_scope="Admission/observed workspace/log caps; not hard OS whole-filesystem/RSS bounds.")
        E.write(route.raw/"environment_result.json",result);status=0
    except BaseException as exc:
        failure={"status":"ENVIRONMENT_PREPARATION_FAILED_OR_INTERRUPTED_NO_RETRY","exception_type":type(exc).__name__,"message":str(exc),"native_use_review_clear":False,"native_execution_authorized":False}
        if hasattr(exc,"child_receipt"):failure["known_child_wait"]=exc.child_receipt
        E.write(route.raw/"failure.json",failure)
    finally:
        own=S.resource.getrusage(S.resource.RUSAGE_SELF)
        E.write(route.raw/"controller_pre_finalization_cost.json",{"controller_interval_seconds":time.perf_counter()-started,"cumulative_own_user_CPU_seconds":own.ru_utime,"cumulative_own_system_CPU_seconds":own.ru_stime,"cumulative_own_peak_RSS_bytes":int(own.ru_maxrss)*1024,"scope":"Linux own cumulative CPU/RSS; interval starts after source admission/Route creation and ends before final ledger/copy inventory. Actual Action step/job wall includes Python startup, admission, all phases and finalization and must be observed separately. Child cumulative peaks are not added."})
        E.write(route.raw/"explicit_transfer_ledger.json",{"requests":route.fetch.rows,"actual_body_bytes":route.fetch.bytes,"pip_resolver_transfer_bytes":"UNKNOWN_NOT_INSTRUMENTED"})
        if not route.packet():status=2
    return status

def artifact_admission(metadata):
    """Size/type gate only, not environment success or evidence interpretation."""
    result={"metadata":False,"wheels":False}
    if metadata is not None and metadata.is_absolute() and metadata.is_dir() and not metadata.is_symlink():
        result["metadata"]=not any(p.is_symlink() for p in metadata.rglob("*")) and E.size(metadata)<=E.CAP["metadata"]
    wheels=TASK/"wheel_artifact"
    if wheels.is_dir() and (wheels/"wheel_manifest.json").is_file():
        result["wheels"]=not any(p.is_symlink() for p in wheels.rglob("*")) and E.size(wheels)<=256*E.MIB
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--source-gate",action="store_true");p.add_argument("--artifact-admission",action="store_true")
    p.add_argument("--metadata-dir",type=Path);p.add_argument("--github-output",type=Path)
    a=p.parse_args()
    if a.artifact_admission:
        flags=artifact_admission(a.metadata_dir)
        if a.github_output:
            with a.github_output.open("a") as out:out.write("".join(k+"="+str(v).lower()+"\n" for k,v in flags.items()))
        print(json.dumps({"artifact_size_admission_only":flags,"native_use_review_clear":False}));return 0
    if not a.execute and not a.source_gate:print(json.dumps({"status":"HELD_ENVIRONMENT_ROUTE_NO_PACKAGE_ACTION","native_use_review_clear":False}));return 0
    repo=HERE.parents[1].resolve()
    def git(*v):return subprocess.check_output(["git","-C",str(repo),*v],text=True).strip()
    head=git("rev-parse","HEAD")
    if head!=os.environ.get("GITHUB_SHA") or git("status","--porcelain","--untracked-files=all"):raise ValueError("exact clean event-SHA checkout required")
    registration=E.read(HERE/"RUN_REGISTRATION_01.json")
    go,_=source_gate(repo,E.read(Path(os.environ["GITHUB_EVENT_PATH"])),os.environ,registration)
    if go is not None:activation_gate(head,git("rev-list","--parents","-n","1","HEAD").split(),git("diff","--name-only",registration["source_bundle_commit"],"HEAD").splitlines(),registration)
    if a.source_gate:
        if a.github_output:
            with a.github_output.open("a") as out:out.write("prepare="+("true" if go else "false")+"\n")
        print(json.dumps({"preparation_authorized":go is not None,"native_execution_authorized":False}));return 0
    if go is None:raise ValueError("held registration cannot prepare an environment")
    if a.metadata_dir is None or not a.metadata_dir.is_absolute() or a.metadata_dir.is_symlink() or a.metadata_dir.resolve().is_relative_to(repo):raise ValueError("separate absolute metadata artifact directory required")
    return prepare(go,a.metadata_dir.resolve())

if __name__=="__main__":
    def stopped(number,frame):raise InterruptedError("parent received "+signal.Signals(number).name)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    raise SystemExit(main())
