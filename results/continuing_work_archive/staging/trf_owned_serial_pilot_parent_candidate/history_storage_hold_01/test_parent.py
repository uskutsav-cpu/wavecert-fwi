"""Tiny stdlib dummy-process controls. No actual worker/numerical/PDE imports."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("held_serial_pilot_controls",HERE/"serial_pilot.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def small_plan():
    value=P.bounded_json(HERE/"PILOT_PLAN.json")
    value["caps"].update(complete_process_seconds=.6,poll_seconds=.005,TERM_grace_seconds=.04,
        KILL_grace_seconds=.2,cleanup_poll_seconds=.005,per_child_log_bytes=256)
    return value


def dummy_command(source="import time;time.sleep(.04);print('dummy child returned')"):
    return [sys.executable,"-B","-I","-c",source]


def dummy_validator(directory,slot,contract):
    return {"case_binding":{"manufactured_common_binding":True},"method_status":"manufactured_dummy",
        "unknown_work":False,"worker_result_sha256":"dummyonly","returned_payload_sha256":"dummyonly"}


def pair(path,plan=None,command=None,validator=dummy_validator):
    plan=small_plan() if plan is None else plan
    output=path/"packet"
    source={"manufactured_stdlib_only":True}
    def built(directory,request):
        # Actual disk state proves the whole fixed pair/source/GO/plan preceded
        # each Popen. This is deliberately not a real project GO/admission.
        for name in ("project_plan.json","supplied_root_go.json","admission.json","source_before.json","attempts.json"):
            if not (output/name).is_file():raise AssertionError("wholepair plan missing before Popen")
        if request["absolute_started"]>time.perf_counter():raise AssertionError("future origin")
        return dummy_command() if command is None else command(directory,request)
    return P.run_pair(output,plan,{"manufactured_only":True,"execution_authorized":False},
        {"manufactured_only":True},source,built,lambda:source,validator,dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))


class ParentControls(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="trf-parent-dummy-")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def test_01_default_actual_isolated_process_is_metadata_without_numerical_imports(self):
        script="""import runpy,sys
sys.argv=[sys.argv[1]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:assert exc.code==0
assert not any(k=='numpy' or k.startswith(('numpy.','scipy','wavecert')) for k in sys.modules)
"""
        result=subprocess.run([sys.executable,"-B","-I","-c",script,str(HERE/"serial_pilot.py")],
            capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)["project_execution_authorized"])

    def test_02_pure_GO_values_reject_full48_wrong_source_missing_review_and_bool_slots(self):
        plan=small_plan();contract=P.parent_contract()
        roles={name:"a"*64 for name in ("auditor","controls","parent_review","auditor_review","runtime_review")}
        roles.update(criteria=plan["criteria_sha256"],worker_review=plan["worker_review_sha256"])
        sources={"worker_contract":plan["worker_contract"],"parent_contract":contract,"role_bindings":roles}
        # Pure in-memory manufactured admission values; never saved as a GO.
        go={"authorization":P.AUTHORIZATION,"mode":plan["mode"],"allowed_slot_indices":[0,1],
            "full48_authorized":False,"source_files":plan["worker_contract"],
            "production_source_sha256":plan["production_source_sha256"],"supervisor_source_sha256":contract["serial_pilot.py"],
            "parent_contract":contract,"admission_sha256":"b"*64,"review_bindings":roles,
            "auditor_source_sha256":roles["auditor"],"audit_criteria_sha256":plan["criteria_sha256"]}
        for name in ("execution_authorized","independent_worker_source_review_clear","external_supervisor_review_clear",
                "independent_auditor_review_clear","runtime_review_clear","mandatory_frozen_controls_passed"):go[name]=True
        P.validate_go(go,plan,sources,"b"*64)
        for key,value in (("full48_authorized",True),("allowed_slot_indices",[False,True]),
                ("allowed_slot_indices",[0,1,2]),("source_files",{}),("independent_auditor_review_clear",False),
                ("mandatory_frozen_controls_passed",False),("supervisor_source_sha256","c"*64)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                P.validate_go({**go,key:value},plan,sources,"b"*64)
        with self.assertRaises(ValueError):P.bound_file(self.root,{"path":"../escape","sha256":"a"*64})

    def test_03_two_serial_dummy_children_pay_earlier_origin_and_have_closed_stable_inventory(self):
        code,report=pair(self.root)
        self.assertEqual(code,0)
        rows=report["attempts"];self.assertEqual([r["slot"]["run_index"] for r in rows],[0,1])
        for row in rows:
            self.assertEqual(row["actual_wait_returncode"],0)
            self.assertEqual(row["actual_pid"],row["observed_pgid"])
            self.assertTrue(row["wait_observed"] and row["terminal_state_known"] and row["closed_log"] and row["eligible"])
            self.assertFalse(row["owned_process_group_alive_after_cleanup"])
            self.assertGreaterEqual(row["external_complete_return_seconds"],row["external_wait_cleanup_logclose_seconds"])
            request=P.bounded_json(self.root/"packet"/("slot_%02d"%row["slot"]["run_index"])/"request.json")
            self.assertEqual(request["absolute_started"],row["absolute_started"])
            self.assertEqual(request["parent_pid"],os.getpid())
        self.assertGreaterEqual(rows[1]["absolute_started"],rows[0]["external_return_observed_perf_counter"])
        inventory=P.bounded_json(self.root/"packet/artifact_inventory.json")
        self.assertTrue(inventory["complete_inventory"])
        self.assertFalse(report["parent_terminal_externally_observed"])
        for relative,row in inventory["files"].items():
            self.assertEqual(P.digest(self.root/"packet"/relative),row["sha256"])

    def test_04_negative_actual_exit_stops_second_slot_without_retry(self):
        command=lambda *unused:dummy_command("import os,signal,time;time.sleep(.04);os.kill(os.getpid(),signal.SIGTERM)")
        code,report=pair(self.root,command=command)
        self.assertEqual(code,2)
        first,second=report["attempts"]
        self.assertEqual(first["actual_wait_returncode"],-signal.SIGTERM)
        self.assertTrue(first["wait_observed"] and first["terminal_state_known"])
        self.assertFalse(first["eligible"])
        self.assertEqual(second["status"],"unattempted")
        self.assertFalse((self.root/"packet/slot_01").exists())

    def test_05_absolute_deadline_reaps_TERM_ignoring_child_and_preserves_negative_KILL(self):
        plan=small_plan();plan["caps"]["complete_process_seconds"]=.12
        command=lambda *unused:dummy_command("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(3)")
        code,report=pair(self.root,plan=plan,command=command)
        row=report["attempts"][0]
        self.assertEqual(code,2);self.assertEqual(row["actual_wait_returncode"],-signal.SIGKILL)
        self.assertFalse(row["eligible"] or row["owned_process_group_alive_after_cleanup"])
        self.assertGreater(row["external_complete_return_seconds"],plan["caps"]["complete_process_seconds"])
        self.assertEqual(report["attempts"][1]["status"],"unattempted")

    def test_06_real_signal_at_Popen_handoff_and_repeats_during_cleanup_are_deferred(self):
        actual_spawn=P.subprocess.Popen;actual_cleanup=P.cleanup_owned
        def handoff(*args,**kwargs):
            child=actual_spawn(*args,**kwargs)
            os.kill(os.getpid(),signal.SIGTERM)  # real signal before outer STORE_FAST
            return child
        def repeated(*args,**kwargs):
            os.kill(os.getpid(),signal.SIGINT);os.kill(os.getpid(),signal.SIGTERM)
            return actual_cleanup(*args,**kwargs)
        with patch.object(P.subprocess,"Popen",handoff),patch.object(P,"cleanup_owned",repeated):
            code,report=pair(self.root,command=lambda *unused:dummy_command("import time;time.sleep(3)"))
        row=report["attempts"][0]
        self.assertEqual(code,2)
        self.assertEqual(row["deferred_parent_signals"],["SIGTERM","SIGINT","SIGTERM"])
        self.assertTrue(row["wait_observed"] and row["terminal_state_known"] and row["closed_log"])
        self.assertFalse(row["owned_process_group_alive_after_cleanup"] or row["eligible"])
        self.assertTrue((self.root/"packet/artifact_inventory.json").exists())
        # A separate real stop between serialized run_slot return and caller
        # assignment must also prevent admission of slot1.
        later=self.root/"return-handoff";later.mkdir();actual_slot=P.run_slot
        def returned(*args,**kwargs):
            value=actual_slot(*args,**kwargs);os.kill(os.getpid(),signal.SIGTERM);return value
        with patch.object(P,"run_slot",returned):code,report=pair(later)
        self.assertEqual(code,2)
        self.assertEqual(report["attempts"][0]["parent_signals_outside_slot"],["SIGTERM"])
        self.assertEqual(report["attempts"][1]["status"],"unattempted")

    def test_07_postwait_receipt_IO_error_keeps_actual_exit_and_stops_later_admission(self):
        actual=P.write_new
        def failed(path,value):
            if Path(path).name=="actual_wait.json":raise OSError("manufactured postwait receipt failure")
            return actual(path,value)
        with patch.object(P,"write_new",failed):code,report=pair(self.root)
        row=report["attempts"][0]
        self.assertEqual(code,2);self.assertEqual(row["actual_wait_returncode"],0)
        self.assertTrue(row["terminal_state_known"] and row["wait_observed"] and row["closed_log"])
        self.assertFalse(row["eligible"])
        self.assertEqual(row["post_wait_IO_errors"][0]["phase"],"actual_wait_receipt")
        saved=P.bounded_json(self.root/"packet/attempts.json")
        self.assertEqual(saved[0]["actual_wait_returncode"],0)
        self.assertEqual(saved[1]["status"],"unattempted")

    def test_08_ambiguous_nohandle_spawn_never_grants_stable_child_hashes(self):
        with patch.object(P.subprocess,"Popen",side_effect=OSError("manufactured unknown launch outcome")):
            code,report=pair(self.root)
        row=report["attempts"][0]
        self.assertEqual(code,2);self.assertEqual(row["launch_state"],"POPEN_CONSTRUCTION_STARTED_OUTCOME_UNKNOWN")
        self.assertIsNone(row["actual_pid"])
        self.assertFalse(row["terminal_state_known"] or row["eligible"])
        self.assertTrue((self.root/"packet/UNSTABLE_CHILD_PREFIX_ONLY.json").exists())
        self.assertFalse((self.root/"packet/artifact_inventory.json").exists())

    def test_09_log_cap_and_worker_unknown_policy_late_flags_are_rejected(self):
        command=lambda *unused:dummy_command("import time;print('x'*4096,flush=True);time.sleep(3)")
        code,report=pair(self.root,command=command)
        row=report["attempts"][0]
        self.assertEqual(code,2);self.assertIn("log cap",row["message"])
        self.assertTrue(row["terminal_state_known"])
        self.assertEqual(report["attempts"][1]["status"],"unattempted")
        directory=self.root/"fixture";worker=directory/"worker";worker.mkdir(parents=True)
        slot=small_plan()["slots"][0];contract={"manufactured_only":"not_real_worker"}
        binding={"case_index":0,**{k:"a"*64 for k in ("raw24_m_sha256","initial49_m_sha256","truth49_m_sha256","all8_observations_sha256","survey_sha256")}}
        result={"status":"WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT","source_files":contract,
            "slot":slot,"method_policy_valid":True,"API_return_gate_only":{"eligible":True},
            "complete_OS_process_eligibility_available":False,"method_status":"solver_success","case_binding":binding}
        payload={"arm":slot["arm"],"policy_valid":True,"unknown_work":False,"preparation_work_unknown":False,
            "error":None,"selected":{"elapsed_seconds":.2},"selected_model":[.2],"status":"solver_success"}
        P.write_new(worker/"result.json",result);P.write_new(worker/"returned_payload.json",payload)
        P.read_worker_result(directory,slot,contract)
        for key,value in (("policy_valid",False),("unknown_work",True),("preparation_work_unknown",True),
                ("selected",{"elapsed_seconds":7.5001})):
            changed=copy.deepcopy(payload);changed[key]=value;P.replace_own(worker/"returned_payload.json",changed)
            with self.subTest(key=key),self.assertRaises(ValueError):P.read_worker_result(directory,slot,contract)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",type=Path,required=True)
    args=parser.parse_args();suite=unittest.defaultTestLoader.loadTestsFromTestCase(ParentControls)
    ids=[test.id() for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
        "test_ids":ids,"source_files":P.parent_contract(),"dummy_children_only":True,"numerical_imports":0,
        "native_owners":0,"PDE_calls":0,"real_worker_executed":False,"GO_created":False}
    P.write_new(args.result,report)
    print(json.dumps(report,sort_keys=True))
    return 0 if result.wasSuccessful() and not result.skipped and result.testsRun==9 else 2


if __name__=="__main__":raise SystemExit(main())
