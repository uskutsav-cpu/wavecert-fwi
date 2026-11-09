"""Stdlib environment-child lifecycle adapted from frozen3f5; no actions on import.
Delta: DEVNULL stdin, optional observer,1MiB log cap, attached wait on receipt I/O.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path
LOG_CAP=1024**2

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())

class EvidenceFailure(RuntimeError):
    def __init__(self,row,cause):
        super().__init__(str(cause));self.child_receipt=row

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


def run_child(label,command,*,cwd,raw,env,timeout,observer=None):
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
                process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                row.update(actual_pid=process.pid,launch_state="HANDLE_ASSIGNED")
                try:row["observed_pgid"]=os.getpgid(process.pid)
                except ProcessLookupError:pass
                if row["observed_pgid"]==process.pid:pgid=process.pid
                write(raw/(label+"_spawned.json"),row)
                if pgid is None:raise ValueError("owned new-session group identity not known")
                while True:
                    if observer is not None:observer(process,pgid)
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
        try:
            write(raw/(label+"_actual_wait.json"),row)
        except BaseException as exc:
            raise EvidenceFailure(row,exc) from exc
        if pending and interrupted is None:interrupted=InterruptedError("parent stop request: "+",".join(pending))
    if interrupted is not None:raise interrupted
    if row.get("owned_process_group_alive_after_cleanup") is not False:raise ValueError("owned child group cleanup not known complete")
    return row

