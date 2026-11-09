"""Manufactured worker/schema/admission only; never imports native9f or runs PDE."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("held_case_worker_controls",HERE/"realcase_worker.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
spec = importlib.util.spec_from_file_location("held_case_runtime_controls",HERE/"case_runtime.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def protocol():
    return json.loads((HERE/"project_protocol.json").read_bytes())


def test_default_actual_isolated_process_has_no_numerical_or_native_import():
    script = """import runpy,sys,json
sys.argv=[sys.argv[1]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:assert exc.code==0
assert not any(k.split('.')[0] in ('numpy','scipy','wavecert') for k in sys.modules)
print(json.dumps({'actual_module_inventory':'NO_NUMERICAL_OR_NATIVE_IMPORTS'}))
"""
    result = subprocess.run([sys.executable,"-B","-I","-c",script,str(HERE/"realcase_worker.py")],
        text=True,capture_output=True,timeout=10,check=False)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["project_execution_authorized"] is False
    assert rows[1]["actual_module_inventory"] == "NO_NUMERICAL_OR_NATIVE_IMPORTS"


def test_actual_source_gate_preserves_old_API_and_adds_project_admission():
    p = protocol()
    repo = HERE.parents[1]/"wavecert-fwi"
    result = W.source_gate(repo.resolve(),p)
    assert result["production_files"] == 161 and result["historical_API_files"] == 38
    wrong = copy.deepcopy(p)
    wrong["observation_blocks"] = 24
    with pytest.raises(ValueError,match="numerical setting"):
        W.source_gate(repo.resolve(),wrong)


def admission_values():
    p, files = protocol(), W.contract()
    slot = json.loads((HERE/p["schedule_path"]).read_bytes())[0]
    go = {"authorization":p["root_go_authorization"],"mode":"case0_two_return_integration_pilot",
        "allowed_slot_indices":[0,1],"full48_authorized":False,"source_files":files,
        "production_source_sha256":p["production_source_sha256"],
        **{key:True for key in ("execution_authorized","independent_worker_source_review_clear",
            "external_supervisor_review_clear","independent_auditor_review_clear","runtime_review_clear")},
        **{key:"a"*64 for key in ("supervisor_source_sha256","auditor_source_sha256","audit_criteria_sha256")}}
    request = {"slot_index":0,"absolute_started":100.,"parent_pid":123,"clock":"time.perf_counter_same_host",
        "optimizer_absolute_deadline":107.5,"complete_return_absolute_deadline":108.,
        "source_contract_sha256":hashlib.sha256(W.canonical(files)).hexdigest(),
        "case_index":0,"arm":slot["arm"]}
    return p,files,go,request


def test_exact_pilot_GO_and_earlier_parent_origin_fail_closed():
    p,files,go,request = admission_values()
    admitted = W.admit_values(go,request,p,files,now=101.,parent_pid=123)
    assert admitted["absolute_started"] == 100. and admitted["slot"]["case_index"] == 0
    for key,value in (("execution_authorized",False),("allowed_slot_indices",list(range(48))),
            ("full48_authorized",True),("independent_auditor_review_clear",False),("source_files",{})):
        with pytest.raises(ValueError):
            W.admit_values({**go,key:value},request,p,files,now=101.,parent_pid=123)
    for key,value in (("slot_index",2),("slot_index",True),("parent_pid",124),
            ("absolute_started",102.),("optimizer_absolute_deadline",109.),("arm","other")):
        with pytest.raises(ValueError):
            W.admit_values(go,{**request,key:value},p,files,now=101.,parent_pid=123)
    with pytest.raises(TimeoutError):
        W.admit_values(go,request,p,files,now=107.5,parent_pid=123)


class NumpyWithoutCachedFields:
    def __getattr__(self,name):
        return getattr(np,name)

    def load(self,*args,**kwargs):
        archive = np.load(*args,**kwargs)
        class Archive:
            files = archive.files
            def __enter__(self):return self
            def __exit__(self,*unused):archive.close()
            def __getitem__(self,key):
                assert key != "wavefields", "cached24grid fields must never be decoded"
                return archive[key]
        return Archive()


class ManufacturedPhysics:
    fail_on_dispatch = None
    def __init__(self,**kwargs):
        self.counts = {"factorizations":0,"forward_rhs":0,"stability_rhs":0}
        self.dispatches = 0

    def cost_snapshot(self):return dict(self.counts)

    def source(self,index):
        value = np.zeros(2401,dtype=np.complex128)
        value[index] = 1600.
        return value

    def solve_state(self,model,rhs,frequency):
        self.dispatches += 1
        self.counts["factorizations"] += 1
        if self.dispatches == self.fail_on_dispatch:
            raise RuntimeError("manufactured interrupted observation dispatch")
        self.counts["forward_rhs"] += rhs.shape[1]
        return np.full((2401,6),complex(frequency,.1),dtype=np.complex128)

    def restrict(self,state,receivers):return state[list(receivers)]


def manufactured_inputs(tmp_path):
    p = protocol()
    repo = tmp_path/"repo"
    directory = repo/"data/reference/subsurfacegen_parent_split"
    directory.mkdir(parents=True)
    base = tmp_path/"worker"
    base.mkdir()
    models = np.stack([np.full((24,24),.2+.01*c,dtype=np.float32) for c in range(8) for _ in range(24)])
    sources = np.tile(np.repeat(p["original_source_indices"],4),8).astype(np.int64)
    frequencies = np.tile(p["original_frequencies_hz"],48).astype(np.float32)
    np.savez_compressed(directory/"validation.npz",models=models,source_indices=sources,frequencies_hz=frequencies,
        receiver_indices=np.arange(50,70,dtype=np.int64),shape=np.array([24,24]),spacing=np.array(.05),
        wavefields=np.array([],dtype=np.float32))  # Unused member; spy forbids even decoding it.
    metadata,split = [],[]
    for c,case in enumerate(p["cases"]):
        identity = case["identity"]
        case["raw_float64_sha256"] = R.array_sha(np,models[24*c].astype(np.float64),"<f8")
        for i in range(24):
            metadata.append({"split":"validation","slice_id":identity["slice_id"],"model_id":identity["parent_id"],
                "source_file":identity["source_file"],"source_sha256":identity["source_sha256"]})
        split.append({"split":"validation","slice_id":identity["slice_id"],"model_id":identity["parent_id"]})
    for name,rows in (("validation_metadata.csv",metadata),("split_manifest.csv",split)):
        with (directory/name).open("w") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    p["inputs_sha256"] = {"data/reference/subsurfacegen_parent_split/"+name:W.sha(directory/name)
        for name in ("validation.npz","validation_metadata.csv","split_manifest.csv")}
    initial = np.stack([np.full(2401,.3+.001*c,dtype=np.float64) for c in range(8)])
    np.savez_compressed(base/"initials.npz",initial_m=initial,case_indices=np.arange(8),shape=np.array([49,49]))
    p["initializer_input"].update(path="initials.npz",sha256=W.sha(base/"initials.npz"))
    for c,case in enumerate(p["cases"]):case["initial_float64_sha256"] = R.array_sha(np,initial[c],"<f8")
    slot = json.loads((HERE/"api/FROZEN48_SCHEDULE.json").read_bytes())[0]
    accepted = {"slot":slot,"absolute_started":100.,"source_files":W.contract()}
    runtime = SimpleNamespace(np=NumpyWithoutCachedFields(), interpolation=RegularGridInterpolator,
        helm=SimpleNamespace(Helmholtz2D=ManufacturedPhysics,SurveyBlock=lambda *args:SimpleNamespace(
            source_index=args[0],frequency_hz=args[1],observed=args[2],label=args[3])),
        driver=SimpleNamespace(PolicyInput=lambda **kwargs:SimpleNamespace(**kwargs)))
    output = tmp_path/"output"
    output.mkdir()
    gate = W.Gate(output,p,clock=lambda:100.)
    gate.base = base
    return p,SimpleNamespace(repo=repo),accepted,runtime,gate


def test_manufactured_schema_public_batches_truth_free_policy_and_new_bindings(tmp_path):
    p,args,accepted,runtime,gate = manufactured_inputs(tmp_path)
    private,attempt = {},{}
    problem = R.prepare_case(args,p,accepted,gate,runtime,private,attempt,clock=lambda:100.)
    assert len(problem.blocks) == 48 and len(problem.receivers) == 20
    assert not hasattr(problem,"truth") and not hasattr(problem,"raw_model")
    assert problem.preparation_costs == {"factorizations":8,"forward_rhs":48,"stability_rhs":0}
    assert attempt["preparation_prefix"]["observation_dispatches_completed"] == 8
    assert not attempt["preparation_prefix"]["unknown_work"]
    assert (gate.output/"prepared_models.npz").exists() and (gate.output/"observations.npz").exists()
    binding = private["binding"]
    W.require_same_case_binding(binding,dict(binding))
    for key in ("initial49_m_sha256","truth49_m_sha256","all8_observations_sha256"):
        with pytest.raises(ValueError,match="prepared-case bytes"):
            W.require_same_case_binding(binding,{**binding,key:"different"})


def test_failed_public_batch_preserves_actual_manufactured_prefix_unknown_tail_and_partial_data(tmp_path):
    p,args,accepted,runtime,gate = manufactured_inputs(tmp_path)
    attempt = {}
    with patch.object(ManufacturedPhysics,"fail_on_dispatch",3),pytest.raises(RuntimeError):
        R.prepare_case(args,p,accepted,gate,runtime,{},attempt,clock=lambda:100.)
    prefix = attempt["preparation_prefix"]
    assert prefix["observation_dispatches_attempted"] == 3 and prefix["observation_dispatches_completed"] == 2
    assert prefix["last_known_observation_counts"] == {"factorizations":3,"forward_rhs":12,"stability_rhs":0}
    assert prefix["unknown_work"] and prefix["observation_call_inflight"]
    with np.load(gate.output/"partial_observations.npz",allow_pickle=False) as saved:
        assert saved["observations"].shape == (12,20)
        assert saved["block_indices"].tolist() == list(range(12))


def test_wrong_raw_case_or_retained_initial_fails_before_fake_observation_owner(tmp_path):
    p,args,accepted,runtime,gate = manufactured_inputs(tmp_path)
    p["cases"][0]["raw_float64_sha256"] = "wrong"
    with pytest.raises(ValueError,match="raw24"):
        R.decode_inputs(args.repo,gate.base,p,accepted["slot"],runtime)
    p["cases"][0]["raw_float64_sha256"] = R.array_sha(np,np.full((24,24),np.float32(.2)).astype(np.float64),"<f8")
    p["cases"][0]["initial_float64_sha256"] = "wrong"
    with pytest.raises(ValueError,match="initializer"):
        R.decode_inputs(args.repo,gate.base,p,accepted["slot"],runtime)


def test_lifecycle_failure_keeps_all_prefix_branches_without_numerical_runner(tmp_path):
    p = protocol()
    args = SimpleNamespace(output=tmp_path/"new-output",repo=HERE.parents[1]/"wavecert-fwi")
    def admitted(*args,**kwargs):return {"source_files":W.contract(),"manufactured_admission_only":True}
    def failing(args,p,accepted,gate):
        gate.attempt.update(preparation_prefix={"known":1},component_prefix={"known":2},
            normalization_prefix={"known":3},engine_pending_prefix={"unknown":True},
            return_payload_before_serialization={"model":np.ones(2)})
        raise OSError("manufactured outer serialization failure")
    assert W.lifecycle(args,p,admission=admitted,limiter=lambda p:None,runner=failing,clock=lambda:100.) == 2
    value = json.loads((args.output/"failure.json").read_bytes())
    assert set(value["evidence"]["attempt"]) == {"preparation_prefix","component_prefix","normalization_prefix",
        "engine_pending_prefix","return_payload_before_serialization"}
    assert value["evidence"]["attempt"]["return_payload_before_serialization"]["model"]["array_shape"] == [2]
    assert "wavecert.physics.helmholtz" not in sys.modules
    # Initial receipt failure must be caught too, without a failing re-query.
    early = SimpleNamespace(output=tmp_path/"early-contract-failure",repo=args.repo)
    actual_write = W.write_new
    def early_write(path,value):
        if Path(path).name == "source_contract_before.json":
            raise OSError("manufactured initial contract receipt failure")
        return actual_write(path,value)
    with patch.object(W,"write_new",early_write),patch.object(W,"contract",wraps=W.contract) as queried:
        assert W.lifecycle(early,p,admission=admitted,limiter=lambda p:None,runner=failing,clock=lambda:100.) == 2
        assert queried.call_count == 1
    failed = json.loads((early.output/"failure.json").read_bytes())
    assert failed["source_files_last_observed"]
    assert failed["current_source_contract_not_requeried_after_failure"]


def test_quality_is_only_called_after_API_returned_immutable_selection(tmp_path):
    p,args,accepted,runtime,gate = manufactured_inputs(tmp_path)
    events = []
    def fake_execute(**kwargs):
        kwargs["admit"]()
        problem = kwargs["prepare"](kwargs["attempt"])
        selected = problem.initial_model*.97
        events.append("selection_returned")
        return {"serialized":b'{"manufactured":true}\n',"selected_model":selected,
            "initial_model":problem.initial_model,"payload":{"status":"manufactured","policy_valid":True}}
    runtime.driver.execute_one = fake_execute
    runtime.driver.source_admission = lambda *args:{"manufactured_source_admission":True}
    runtime.driver.observe_return = lambda *args:{"eligible":True,"scope":"manufacturedAPIonly"}
    runtime.api = HERE/"api"
    runtime.origins,runtime.manufactured_runtime = {"manufactured":"no native imports"},True
    gate.project_source_check = lambda:None
    original = R.quality_after_return
    def quality(*args):
        assert events == ["selection_returned"]
        events.append("quality_after_return")
        return original(*args)
    with patch.object(R,"quality_after_return",quality):
        result = R.run_case(args,p,accepted,gate,runtime_factory=lambda *args:runtime,clock=lambda:100.)
    assert events == ["selection_returned","quality_after_return"]
    assert result["complete_OS_process_eligibility_available"] is False
    assert result["post_return_quality"]["used_for_selection"] is False
    # A secondary failure saving actual available vectors cannot mask the
    # original API interruption or erase its available prefix.
    failed_output = tmp_path/"failed-model-save"
    failed_output.mkdir()
    failed_gate = W.Gate(failed_output,p,clock=lambda:100.)
    failed_gate.base,failed_gate.project_source_check = gate.base,lambda:None
    def interrupted_execute(**kwargs):
        kwargs["attempt"]["initial_model"] = np.ones(2)
        raise KeyboardInterrupt("manufactured original API interruption")
    runtime.driver.execute_one = interrupted_execute
    actual_save = R.save_arrays
    def failed_save(np_,path,**arrays):
        if Path(path).name == "failure_models.npz":
            raise OSError("manufactured secondary persistence failure")
        return actual_save(np_,path,**arrays)
    with patch.object(R,"save_arrays",failed_save),pytest.raises(KeyboardInterrupt,match="original API"):
        R.run_case(args,p,accepted,failed_gate,runtime_factory=lambda *args:runtime,clock=lambda:100.)
    assert failed_gate.context["original_case_exception"]["exception_type"] == "KeyboardInterrupt"
    assert failed_gate.context["failure_model_persistence_error"]["exception_type"] == "OSError"
    assert failed_gate.context["failure_model_persistence_error"]["available_vectors_remain_in_attempt_prefix"]
