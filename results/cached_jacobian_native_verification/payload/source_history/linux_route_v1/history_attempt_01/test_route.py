"""Stdlib-only route controls: manufactured events and actual tiny dummy PIDs."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("held_route_controls",HERE/"run_route.py")
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class RouteControls(unittest.TestCase):
    def context(self):
        head="a"*40
        event={"repository":{"full_name":R.REPOSITORY,"private":False,"visibility":"public"},
            "ref":"refs/heads/"+R.BRANCH,"created":True,"forced":False,"deleted":False,"before":"0"*40,
            "after":head,"head_commit":{"id":head}}
        env={"GITHUB_EVENT_NAME":"push","GITHUB_REPOSITORY":R.REPOSITORY,"GITHUB_REF":event["ref"],
            "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":head,"GITHUB_WORKFLOW_SHA":head,
            "GITHUB_WORKFLOW_REF":R.REPOSITORY+"/.github/workflows/cached-j-native.yml@"+event["ref"]}
        return event,env,head

    def test_exact_first_push_and_parent_context(self):
        event,env,head=self.context()
        R.event_gate(event,env,head,[head,R.BASELINE],R.BASELINE)
        for key,value in (("created",False),("forced",True),("deleted",True),("before","b"*40)):
            altered=dict(event,**{key:value})
            with self.assertRaises(ValueError):R.event_gate(altered,env,head,[head,R.BASELINE],R.BASELINE)
        with self.assertRaises(ValueError):R.event_gate(event,dict(env,GITHUB_RUN_ATTEMPT="2"),head,[head,R.BASELINE],R.BASELINE)
        with self.assertRaises(ValueError):R.event_gate(event,env,head,[head,"b"*40],R.BASELINE)

    def test_public_repo_branch_and_workflow_guards(self):
        event,env,head=self.context()
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):R.event_gate(event,env,head,[head,R.BASELINE],R.BASELINE)
        event["repository"]["private"]=False
        for key,value in (("GITHUB_REF","refs/heads/main"),("GITHUB_REPOSITORY","foreign/repo"),("GITHUB_WORKFLOW_SHA","c"*40)):
            with self.assertRaises(ValueError):R.event_gate(event,dict(env,**{key:value}),head,[head,R.BASELINE],R.BASELINE)

    def registration(self):
        return {"repository":R.REPOSITORY,"branch":R.BRANCH,"single_parent":R.BASELINE,
            "route_manifest_sha256":"m"*64,"native_execution_registered":False,"root_go":None}

    def test_false_null_is_controls_only_without_extra_mandatory_phase(self):
        value=self.registration()
        self.assertIsNone(R.registration_gate(value,"m"*64))
        value["root_go"]={"not":"authorization"}
        with self.assertRaises(ValueError):R.registration_gate(value,"m"*64)

    def test_conditional_go_can_admit_first58_then_native_after_local_review(self):
        value=self.registration();value["native_execution_registered"]=True
        go={"authorization":"GO_CACHED_J_NATIVE_FIELD_VERIFICATION","execution_authorized":True,
            "independent_source_review_clear":True,"criteria_frozen_before_execution":True,
            "source_files":R.WORKER_CONTRACT,"route_manifest_sha256":"m"*64,
            "route_registration_sha256":R.hashlib.sha256(R.canonical({k:v for k,v in value.items() if k!="root_go"})).hexdigest()}
        value["root_go"]=go
        self.assertEqual(R.registration_gate(value,"m"*64),go)
        value["root_go"]={**go,"route_manifest_sha256":"changed"}
        with self.assertRaises(ValueError):R.registration_gate(value,"m"*64)

    def child(self,code,*,timeout=3.):
        temporary=tempfile.TemporaryDirectory(prefix="cachedj-route-")
        self.addCleanup(temporary.cleanup)
        raw=Path(temporary.name)
        row=R.run_child("dummy",[sys.executable,"-B","-I","-c",code],cwd=raw,raw=raw,env=dict(os.environ),timeout=timeout)
        return row,raw

    def test_actual_pid_group_wait_and_stable_closed_log(self):
        row,raw=self.child('print("manufactured stdlib child")')
        self.assertGreater(row["actual_pid"],0);self.assertEqual(row["observed_pgid"],row["actual_pid"])
        self.assertEqual(row["actual_wait_returncode"],0);self.assertTrue(row["wait_observed"])
        self.assertFalse(row["owned_process_group_alive_after_cleanup"])
        R.artifact_packet(raw,raw/"artifact")
        inventory=json.loads((raw/"artifact/artifact_inventory.json").read_bytes())
        self.assertTrue(inventory["files"]["dummy.log"]["included"])
        self.assertEqual(inventory["files"]["dummy.log"]["sha256"],R.digest(raw/"dummy.log"))

    def test_actual_nonzero_and_negative_signal_exits_not_relabeled(self):
        row,_=self.child('raise SystemExit(7)')
        self.assertEqual(row["actual_wait_returncode"],7)
        row,_=self.child('import os,signal;os.kill(os.getpid(),signal.SIGTERM)')
        self.assertEqual(row["actual_wait_returncode"],-signal.SIGTERM)

    def test_timeout_cleans_known_child_and_retains_actual_wait(self):
        with tempfile.TemporaryDirectory(prefix="cachedj-timeout-") as temporary:
            raw=Path(temporary)
            with self.assertRaises(R.subprocess.TimeoutExpired):
                R.run_child("dummy",[sys.executable,"-B","-I","-c","import time;time.sleep(20)"],cwd=raw,raw=raw,env=dict(os.environ),timeout=.1)
            row=json.loads((raw/"dummy_actual_wait.json").read_bytes())
            self.assertTrue(row["wait_observed"])
            self.assertLess(row["actual_wait_returncode"],0)
            self.assertFalse(row["owned_process_group_alive_after_cleanup"])
            self.assertIn("SIGTERM",row["signals"])

    def test_spawn_failure_retains_prefix_without_fake_pid(self):
        with tempfile.TemporaryDirectory(prefix="cachedj-spawn-") as temporary:
            raw=Path(temporary)
            with self.assertRaises(FileNotFoundError):
                R.run_child("dummy",["/not/an/executable"],cwd=raw,raw=raw,env=dict(os.environ),timeout=1.)
            row=json.loads((raw/"dummy_actual_wait.json").read_bytes())
            self.assertIsNone(row["actual_pid"]);self.assertFalse(row["wait_observed"])
            self.assertTrue((raw/"dummy_requested.json").exists())

    def test_unreaped_child_blocks_stable_inventory(self):
        with tempfile.TemporaryDirectory(prefix="cachedj-unreaped-") as temporary:
            raw=Path(temporary)
            R.write(raw/"dummy_requested.json",{"manufactured":True})
            (raw/"dummy.log").write_text("not a known closed child log")
            with self.assertRaises(ValueError):R.artifact_packet(raw,raw/"artifact")
            self.assertFalse((raw/"artifact/artifact_inventory.json").exists())
            self.assertTrue((raw/"artifact/UNSTABLE_CHILD_PREFIX_ONLY.json").exists())

    def test_cumulative_rss_scope_does_not_assign_prior_peak_to_native(self):
        row,_=self.child('print("scope only")')
        self.assertIn("Cumulative",row["RSS_scope"])
        self.assertNotIn("native_peak_RSS_bytes",row)



if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RouteControls))
    print(json.dumps({"status":"MANUFACTURED_ROUTE_PASS" if result.wasSuccessful() else "FAIL",
        "tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "numerical_imports":0,"native_jobs":0,"remote_jobs":0,"project_arrays":0}))
    raise SystemExit(0 if result.wasSuccessful() else 2)
