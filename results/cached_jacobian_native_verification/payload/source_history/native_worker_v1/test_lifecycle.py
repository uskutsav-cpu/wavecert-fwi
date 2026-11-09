"""Stdlib-only lifecycle controls; no copied numerical component is imported."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("held_field_lifecycle",ROOT/"verify_native_fields.py")
V=importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class LifecycleControls(unittest.TestCase):
    def protocol(self):
        return json.loads((ROOT/"protocol.json").read_bytes())

    def args(self,path):
        return argparse.Namespace(output=path,repo=Path("UNREAD_MANUFACTURED_REPO"),root_go=None)

    def test_default_isolated_cli_blocks_all_numerical_and_native_imports(self):
        wrapper='''import sys,runpy
def block(event,args):
 if event=="import" and args[0].split(".")[0] in {"numpy","scipy","wavecert","cached_jacobian","public_map_builder"}:
  raise RuntimeError("numerical import forbidden by manufactured control")
sys.addaudithook(block)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        with tempfile.TemporaryDirectory(prefix="field-default-") as temporary:
            result=subprocess.run([sys.executable,"-B","-I","-c",wrapper,str(ROOT/"verify_native_fields.py")],cwd=temporary,text=True,capture_output=True,check=False)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"],"HELD_NATIVE_INTEGRATION_SOURCE_ONLY")
            self.assertEqual(list(Path(temporary).iterdir()),[])

    def test_fixed_plan_has_all_terms_and_separately_charged_work(self):
        p=self.protocol()
        self.assertEqual((len(p["frequencies_hz"]),len(p["source_indices"]),len(p["receiver_indices"])),(8,6,20))
        self.assertEqual(2*8*6*20,p["real_residual_terms"])
        self.assertTrue(set(p["source_indices"]).issubset(p["receiver_indices"]))
        # Independent integer reconciliation of the declared public-API work.
        self.assertEqual(p["expected_complete_disjoint_counts"],{
            "factorizations":8+16+8+8+8+8+8+4*8,
            "forward_rhs":48+96+48+48+48+4*48,
            "tangent_rhs":48,"adjoint_rhs":48,"stability_rhs":8*(20-6)+8*20})
        self.assertEqual(p["fd_steps"],[1e-4,5e-5])
        self.assertFalse(p["execution_authorized"])

    def test_plan_is_durable_before_limiter_and_worker(self):
        p=self.protocol();calls=[]
        with tempfile.TemporaryDirectory(prefix="field-plan-") as temporary:
            path=Path(temporary)/"fresh"
            def limiter(protocol):
                self.assertEqual(json.loads((path/"plan.json").read_bytes()),p)
                self.assertTrue((path/"admission.json").is_file())
                self.assertIn("before_numerical_imports",(path/"events.jsonl").read_text())
                calls.append("limiter")
            def worker(args,protocol,gate):
                calls.append("manufactured_worker")
                self.assertEqual(calls,["limiter","manufactured_worker"])
                return {"status":"MANUFACTURED_ONLY_NO_NATIVE_CALL"}
            self.assertEqual(V.lifecycle(self.args(path),p,{"manufactured":True},worker=worker,limiter=limiter),0)
            result=json.loads((path/"result.json").read_bytes())
            self.assertEqual(result["status"],"MANUFACTURED_ONLY_NO_NATIVE_CALL")
            self.assertTrue(result["actual_external_wait_required"])

    def test_limiter_failure_prevents_worker_and_preserves_full_plan(self):
        with tempfile.TemporaryDirectory(prefix="field-limits-") as temporary:
            path=Path(temporary)/"fresh"
            def limiter(p):raise RuntimeError("manufactured resource gate")
            def worker(*args):self.fail("worker must not be reached")
            self.assertEqual(V.lifecycle(self.args(path),self.protocol(),{},worker=worker,limiter=limiter),2)
            self.assertTrue((path/"plan.json").is_file())
            self.assertIn("resource gate",(path/"failure.json").read_text())

    def test_resource_gate_sets_explicit_kernel_limits_without_exercising_them(self):
        p=self.protocol()
        with patch.object(V.platform,"system",return_value="Linux"), \
                patch.object(V.platform,"python_version",return_value=p["runtime"]["python"]), \
                patch.object(V.importlib.metadata,"version",side_effect=lambda name:p["runtime"][name]), \
                patch.object(V.resource,"setrlimit") as setlimit, \
                patch.object(V.signal,"signal") as handler,patch.object(V.signal,"alarm") as alarm, \
                patch.dict(V.os.environ,{},clear=False):
            V.limits(p)
            self.assertEqual(setlimit.call_count,2)
            setlimit.assert_any_call(V.resource.RLIMIT_AS,(2147483648,2147483648))
            setlimit.assert_any_call(V.resource.RLIMIT_CPU,(180,180))
            handler.assert_called_once_with(V.signal.SIGALRM,V.signal.SIG_DFL)
            alarm.assert_called_once_with(180)

    def test_failed_public_dispatch_retains_actual_nonzero_prefix_and_unknown(self):
        with tempfile.TemporaryDirectory(prefix="field-prefix-") as temporary:
            path=Path(temporary)/"fresh";state={"attempts":0,"returned_rhs":0}
            def worker(args,p,gate):
                def action():
                    state["attempts"]+=1
                    raise RuntimeError("manufactured interrupted dispatch")
                gate.run("manufactured_not_native",action,lambda:dict(state))
            self.assertEqual(V.lifecycle(self.args(path),self.protocol(),{},worker=worker,limiter=lambda p:None),2)
            value=json.loads((path/"failure.json").read_bytes())
            event=value["evidence"]["calls"][0]
            self.assertEqual(event["status"],"raised")
            self.assertEqual(event["after"]["actual_last_snapshot"]["attempts"],1)
            self.assertTrue(event["unknown_native_tail"])
            self.assertFalse((path/"result.json").exists())

    def test_snapshot_failure_preserves_previous_prefix_instead_of_zero(self):
        with tempfile.TemporaryDirectory(prefix="field-snapshot-") as temporary:
            path=Path(temporary);gate=V.Gate(path,self.protocol())
            gate.snapshot("manufactured",lambda:{"actual_prefix":7})
            def fail():raise RuntimeError("manufactured snapshot failure")
            value=gate.snapshot("manufactured",fail)
            self.assertEqual(value["actual_last_snapshot"],{"actual_prefix":7})
            self.assertTrue(value["unknown_after_prefix"])

    def test_later_snapshot_failure_cannot_rewrite_earlier_observation(self):
        with tempfile.TemporaryDirectory(prefix="field-prefix-freeze-") as temporary:
            gate=V.Gate(Path(temporary),self.protocol())
            before=gate.snapshot("manufactured",lambda:{"prefix":7})
            def fail():raise RuntimeError("manufactured later snapshot failure")
            after=gate.snapshot("manufactured",fail)
            self.assertFalse(before["unknown_after_prefix"])
            self.assertTrue(after["unknown_after_prefix"])
            self.assertIsNot(before,after)

    def test_keyboard_interrupt_prefix_has_no_retry_or_success(self):
        with tempfile.TemporaryDirectory(prefix="field-interrupt-") as temporary:
            path=Path(temporary)/"fresh";calls=[]
            def worker(args,p,gate):
                calls.append("once")
                gate.event("manufactured_interrupted_prefix",not_a_native_PDE=True)
                raise KeyboardInterrupt
            self.assertEqual(V.lifecycle(self.args(path),self.protocol(),{},worker=worker,limiter=lambda p:None),2)
            self.assertEqual(calls,["once"])
            self.assertIn("KeyboardInterrupt",(path/"failure.json").read_text())

    def test_output_serialization_failure_keeps_work_evidence(self):
        with tempfile.TemporaryDirectory(prefix="field-serialize-") as temporary:
            path=Path(temporary)/"fresh"
            def worker(args,p,gate):
                gate.event("manufactured_complete_work_prefix",not_a_native_PDE=True)
                return {"unserializable":object()}
            self.assertEqual(V.lifecycle(self.args(path),self.protocol(),{},worker=worker,limiter=lambda p:None),2)
            self.assertTrue((path/"failure.json").is_file())
            self.assertIn("complete_work_prefix",(path/"events.jsonl").read_text())

    def test_fresh_output_policy_prevents_any_limiter_or_worker(self):
        with tempfile.TemporaryDirectory(prefix="field-fresh-") as temporary:
            def forbidden(*args):self.fail("existing output must reject first")
            with self.assertRaises(FileExistsError):
                V.lifecycle(self.args(Path(temporary)),self.protocol(),{},worker=forbidden,limiter=forbidden)

    def test_public_dispatch_cap_rejects_before_call(self):
        with tempfile.TemporaryDirectory(prefix="field-dispatch-") as temporary:
            p=self.protocol();p["resources"]["maximum_recorded_public_dispatches"]=1
            gate=V.Gate(Path(temporary),p);calls=[]
            gate.run("manufactured",lambda:calls.append("once"),lambda:{"prefix":0})
            with self.assertRaises(ValueError):gate.run("manufactured",lambda:calls.append("twice"),lambda:{"prefix":0})
            self.assertEqual(calls,["once"])

    def test_invalid_go_stops_before_repo_or_output(self):
        with tempfile.TemporaryDirectory(prefix="field-GO-") as temporary:
            path=Path(temporary);go=path/"MANUFACTURED_REJECTED_GO.json"
            go.write_text(json.dumps({"execution_authorized":False,"source_files":V.contract()}))
            args=self.args(path/"not_created");args.root_go=go
            with self.assertRaises(ValueError):V.admit(args,self.protocol())
            self.assertFalse(args.output.exists())

    def test_foreign_production_manifest_rejects_valid_shaped_manufactured_admission(self):
        with tempfile.TemporaryDirectory(prefix="field-source-") as temporary:
            path=Path(temporary);go=path/"MANUFACTURED_UNEXECUTED_GO.json"
            go.write_text(json.dumps({"authorization":self.protocol()["root_go_authorization"],
                "execution_authorized":True,"independent_source_review_clear":True,
                "criteria_frozen_before_execution":True,"source_files":V.contract()}))
            args=self.args(path/"not_created");args.root_go=go
            with patch.object(V,"source_manifest",return_value={}),self.assertRaises(ValueError):
                V.admit(args,self.protocol())
            self.assertFalse(args.output.exists())


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleControls))
    print(json.dumps({"status":"MANUFACTURED_LIFECYCLE_PASS" if result.wasSuccessful() else "FAIL",
        "tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "numerical_imports":0,"native_owners_constructed":0,"PDE_calls":0,
        "source_files":V.contract(),"control_source_sha256":V.sha(Path(__file__))}))
    return 0 if result.wasSuccessful() else 2


if __name__=="__main__":
    raise SystemExit(main())
