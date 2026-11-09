"""Stdlib manufactured route controls. No package/network/native environment actions."""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,tarfile,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("manufactured_environment_route",HERE/"prepare_environment.py")
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
S,E=D.S,D.E
BLOCKED=[]
def guard(event,args):
    if event=="import" and args[0].split(".")[0] in ("numpy","scipy","devito","pyrevolve","examples"):
        BLOCKED.append(args[0]);raise RuntimeError("runtime import forbidden in manufactured controls")
sys.addaudithook(guard)

def dummy(code):return [sys.executable,"-B","-I","-u","-c",code]

def child(root,code,observer=None):
    return S.run_child("dummy",dummy(code),cwd=root,raw=root,env=os.environ.copy(),timeout=5,observer=observer)

def setup_gate(root):
    repo=root/"repo";here=repo/"research/checkpoint23_environment";here.mkdir(parents=True)
    rows={}
    for name in D.EXPECTED_FILES:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("MANUFACTURED_SOURCE_NOT_EXECUTED\n")
        rows[name]={"bytes":p.stat().st_size,"sha256":S.digest(p)}
    E.write(here/"ROUTE_MANIFEST.json",{"schema":"checkpoint23_environment_route_bundle_v1","files":rows})
    reg={"schema":"checkpoint23_environment_preparation_registration_v1","route_manifest_sha256":S.digest(here/"ROUTE_MANIFEST.json"),"v2_runtime_bundle_sha256":D.V2_BUNDLE,"native_execution_authorized":False,"environment_preparation_registered":False,"root_go":None}
    c={"GITHUB_REPOSITORY":D.REPOSITORY,"GITHUB_REF":"refs/heads/"+D.BRANCH,"GITHUB_EVENT_NAME":"push","GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"1"*40,"GITHUB_WORKFLOW_SHA":"1"*40,"GITHUB_WORKFLOW_REF":D.REPOSITORY+"/.github/workflows/checkpoint23-environment-01.yml@refs/heads/"+D.BRANCH}
    event={"forced":False,"deleted":False,"repository":{"private":False},"after":"1"*40}
    return repo,here,reg,c,event

class RouteControls(unittest.TestCase):
    def test_default_CLI_has_no_files_or_runtime_imports(self):
        code='import sys,runpy\nsys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError("forbidden")) if e=="import" and a[0].split(".")[0] in ("numpy","scipy","devito","pyrevolve") else None)\nsys.argv=[sys.argv[1]]\nrunpy.run_path(sys.argv[0],run_name="__main__")'
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,"-B","-I","-c",code,str(HERE/"prepare_environment.py")],cwd=t,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)["status"],"HELD_ENVIRONMENT_ROUTE_NO_PACKAGE_ACTION")
            self.assertEqual(list(Path(t).iterdir()),[])
    def test_real_dummy_zero_has_actual_wait_PID_group_and_closed_log(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);row=child(root,'import os,json;print(json.dumps({"PID":os.getpid(),"PGID":os.getpgrp()}))')
            printed=json.loads((root/"dummy.log").read_text())
            self.assertEqual(row["actual_wait_returncode"],0);self.assertTrue(row["wait_observed"])
            self.assertTrue(row["terminal_state_known"]);self.assertFalse(row["owned_process_group_alive_after_cleanup"])
            self.assertEqual(printed,{"PID":row["actual_pid"],"PGID":row["actual_pid"]})
    def test_real_dummy_nonzero_is_preserved_once(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);row=child(root,'import sys;print("one-attempt");sys.exit(7)')
            self.assertEqual(row["actual_wait_returncode"],7);self.assertEqual((root/"dummy.log").read_text().count("one-attempt"),1)
    def test_parent_signal_during_Popen_handle_handoff_still_reaps(self):
        real=S.subprocess.Popen
        def interrupted(*a,**kw):
            value=real(*a,**kw);signal.raise_signal(signal.SIGTERM);return value
        with tempfile.TemporaryDirectory() as t,patch.object(S.subprocess,"Popen",side_effect=interrupted):
            root=Path(t)
            with self.assertRaises(InterruptedError):child(root,"import time;time.sleep(30)")
            row=E.read(root/"dummy_actual_wait.json")
            self.assertTrue(row["wait_observed"]);self.assertFalse(row["owned_process_group_alive_after_cleanup"])
            self.assertIn("SIGTERM",row["deferred_parent_signals"])
    def test_spawn_receipt_write_error_reaps_known_child(self):
        original=S.write
        def fail(p,v):
            if p.name.endswith("_spawned.json"):raise OSError("manufactured receipt failure")
            return original(p,v)
        with tempfile.TemporaryDirectory() as t,patch.object(S,"write",side_effect=fail):
            root=Path(t)
            with self.assertRaises(OSError):child(root,"import time;time.sleep(30)")
            row=E.read(root/"dummy_actual_wait.json")
            self.assertTrue(row["terminal_state_known"]);self.assertFalse(row["owned_process_group_alive_after_cleanup"])
    def test_resource_observer_error_reaps_known_child(self):
        def fail(*_):raise ValueError("manufactured observation failure")
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises(ValueError):child(root,"import time;time.sleep(30)",fail)
            self.assertTrue(E.read(root/"dummy_actual_wait.json")["terminal_state_known"])
    def test_actual_wait_is_attached_if_final_receipt_write_fails(self):
        original=S.write
        def fail(p,v):
            if p.name.endswith("_actual_wait.json"):raise OSError("manufactured final write failure")
            return original(p,v)
        with tempfile.TemporaryDirectory() as t,patch.object(S,"write",side_effect=fail):
            with self.assertRaises(S.EvidenceFailure) as caught:child(Path(t),"print('done')")
            self.assertEqual(caught.exception.child_receipt["actual_wait_returncode"],0)
            self.assertTrue(caught.exception.child_receipt["terminal_state_known"])
    def test_final_resource_error_retains_already_known_actual_wait(self):
        original=S.resource.getrusage;calls=[]
        def fail_after_wait(kind):
            calls.append(kind)
            if len(calls)>1:raise OSError("manufactured final resource failure")
            return original(kind)
        with tempfile.TemporaryDirectory() as t,patch.object(S.resource,"getrusage",side_effect=fail_after_wait):
            with self.assertRaises(S.EvidenceFailure) as caught:child(Path(t),"print('done')")
            self.assertEqual(caught.exception.child_receipt["actual_wait_returncode"],0)
            self.assertTrue(caught.exception.child_receipt["wait_observed"])
            self.assertFalse(caught.exception.child_receipt["owned_process_group_alive_after_cleanup"])
    def test_packet_includes_closed_prior_controls_and_excludes_payload_tmp(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/"raw";src.mkdir();out=root/"metadata";out.mkdir()
            (out/"controls.log").write_text("closed controls\n");(src/"phase.log").write_text("closed phase\n")
            (src/"tmp").mkdir();(src/"tmp/payload.synthetic").write_text("SYNTHETIC_PACKAGE_PLACEHOLDER")
            E.copy_packet(src,out,4096);rows=E.read(out/"artifact_inventory.json")
            self.assertEqual(set(rows["files"]),{"controls.log","phase.log"});self.assertGreater(rows["excluded_tmp_bytes"],0)
            self.assertFalse((out/"tmp").exists())
    def test_packet_hashes_actual_copy_and_rejects_mutation_during_copy(self):
        original=E.shutil.copyfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/"raw";src.mkdir();out=root/"metadata";(src/"x.log").write_text("old")
            def changed(a,b):
                a.write_text("different");return original(a,b)
            with patch.object(E.shutil,"copyfile",side_effect=changed),self.assertRaises(ValueError):E.copy_packet(src,out,4096)
            self.assertFalse((out/"artifact_inventory.json").exists())
    def test_archive_traversal_duplicate_and_link_rejected(self):
        for names in [["../escape"],["same","same"]]:
            with self.assertRaises(ValueError):E.safe_names(names)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"source.tar.gz"
            with tarfile.open(p,"w:gz") as f:
                m=tarfile.TarInfo("root/link");m.type=tarfile.SYMTYPE;m.linkname="../escape";f.addfile(m)
            with self.assertRaises(ValueError):E.source_tree(p,root/"out")
            self.assertFalse((root/"escape").exists())
    def test_source_backend_path_rejected_without_import(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/"source.tar.gz";body=b'[build-system]\nrequires=[]\nbuild-backend="foreign.backend"\nbackend-path=["."]\n'
            with tarfile.open(p,"w:gz") as f:
                m=tarfile.TarInfo("root/pyproject.toml");m.size=len(body);f.addfile(m,io.BytesIO(body))
            with self.assertRaises(ValueError):E.source_tree(p,root/"out")
    def test_resolver_admits_only_exact_bound_local_wheel_and_rejects_sdist(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()/"manufactured.whl";p.write_bytes(b"NOT_A_REAL_PACKAGE")
            row={"metadata":{"name":"manufactured","version":"1.0"},"download_info":{"url":p.as_uri(),"archive_info":{"hashes":{"sha256":E.digest(p)}}}}
            report={"version":"1","install":[row]}
            with self.assertRaises(ValueError):E.selections(report,{})
            selected=E.selections(report,{str(p):{"sha256":E.digest(p),"scope":"MANUFACTURED_HASH_CONTEXT_NOT_AUTHENTIC_BUILD_RECEIPT"}})
            self.assertTrue(selected[0]["local"])
            row["download_info"]["url"]="https://files.pythonhosted.org/not-allowed.tar.gz"
            with self.assertRaises(ValueError):E.selections(report,{})
    def test_fetch_mismatch_preserves_closed_prefix_without_retry(self):
        class Response(io.BytesIO):
            status=200;headers={"Content-Length":"5"}
            def geturl(self):return "https://files.pythonhosted.org/manufactured.whl"
        opener=unittest.mock.Mock();opener.open.return_value=Response(b"small")
        with tempfile.TemporaryDirectory() as t,patch.object(E.urllib.request,"build_opener",return_value=opener):
            f=E.Fetch();p=Path(t)/"manufactured.whl"
            with self.assertRaises(ValueError):f.get("https://files.pythonhosted.org/manufactured.whl",p,8,"0"*64,5)
            self.assertEqual(opener.open.call_count,1);self.assertFalse(p.exists())
            self.assertEqual(p.with_name(p.name+".part").read_bytes(),b"small")
            self.assertEqual(f.rows[0]["actual_body_bytes"],5);self.assertFalse(f.rows[0]["completed"])
    def test_held_registration_cannot_start_environment_and_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            repo,here,reg,c,event=setup_gate(Path(t).resolve())
            with patch.object(D,"HERE",here):
                self.assertIsNone(D.source_gate(repo,event,c,reg)[0])
                reg["environment_preparation_registered"]=True
                with self.assertRaises(ValueError):D.source_gate(repo,event,c,reg)
                reg["environment_preparation_registered"]=False
                (repo/next(iter(D.EXPECTED_FILES))).write_text("tampered")
                with self.assertRaises(ValueError):D.source_gate(repo,event,c,reg)
    def test_event_attempt_and_activation_parent_are_exact(self):
        with tempfile.TemporaryDirectory() as t:
            repo,here,reg,c,event=setup_gate(Path(t).resolve());c["GITHUB_RUN_ATTEMPT"]="2"
            with patch.object(D,"HERE",here),self.assertRaises(ValueError):D.source_gate(repo,event,c,reg)
        r={"source_bundle_commit":"a"*40}
        D.activation_gate("b"*40,["b"*40,"a"*40],["research/checkpoint23_environment/RUN_REGISTRATION_01.json"],r)
        with self.assertRaises(ValueError):D.activation_gate("b"*40,["b"*40,"a"*40],["unregistered.py"],r)
    def test_namespace_command_has_explicit_privilege_drops_without_execution(self):
        with patch.object(D.os,"readlink",return_value="net:[MANUFACTURED]"),patch.object(D.os,"getuid",return_value=1001),patch.object(D.os,"getgid",return_value=1001):
            argv=D.Route.namespace(object(),sys.executable,Path("/manufactured/output"),["--dummy-signal"])
        for flag in ("--net","--clear-groups","--no-new-privs","--bounding-set=-all","--inh-caps=-all","--ambient-caps=-all"):self.assertIn(flag,argv)
    def test_artifact_size_gate_does_not_award_environment_success(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()
            with patch.object(E,"size",return_value=E.CAP["metadata"]+1):self.assertFalse(D.artifact_admission(p)["metadata"])
            self.assertTrue(D.artifact_admission(p)["metadata"])

if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RouteControls))
    print(json.dumps({"sentinel":"CHECKPOINT23_ENVIRONMENT_ROUTE_MANUFACTURED_V1","tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),"runtime_import_attempts":BLOCKED,"PDE_calls":0,"package_actions":0,"actual_namespace_probe":False,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and result.testsRun==18 and not BLOCKED else 1)
