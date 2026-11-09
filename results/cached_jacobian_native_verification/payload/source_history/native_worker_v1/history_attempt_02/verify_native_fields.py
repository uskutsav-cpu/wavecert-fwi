"""HELD constant-model field integration worker; default is stdlib metadata only.

No project dataset, optimizer, install, download or study driver is included.
Native calls require a separately reviewed exact source-bound root GO.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import resource
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("verify_native_fields.py", "protocol.json", "FROZEN_PRODUCTION_SOURCE.json",
         "cached_jacobian.py", "public_map_builder.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return {name: sha(HERE/name) for name in FILES}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_manifest(repo):
    repo = Path(repo)
    files = [repo/"pyproject.toml"]+sorted((repo/"src").rglob("*.py"))
    files += sorted(p for p in (repo/"configs").rglob("*") if p.is_file())
    files += sorted(p for p in (repo/"scripts").rglob("*") if p.suffix in {".py", ".sh"})
    files += sorted((repo/".github/workflows").glob("*.yml"))+sorted(repo.glob("requirements*.txt"))
    return {p.relative_to(repo).as_posix(): sha(p) for p in sorted(set(files)) if p.is_file()}


def admit(args, protocol):
    if args.root_go is None or args.repo is None or args.output is None:
        raise ValueError("explicit GO, frozen repo and fresh output required")
    go = json.loads(Path(args.root_go).read_bytes())
    if (go.get("authorization") != protocol["root_go_authorization"]
            or go.get("execution_authorized") is not True
            or go.get("independent_source_review_clear") is not True
            or go.get("criteria_frozen_before_execution") is not True
            or go.get("source_files") != contract()):
        raise ValueError("separate exact reviewed root GO required")
    frozen = json.loads((HERE/"FROZEN_PRODUCTION_SOURCE.json").read_bytes())
    if source_manifest(args.repo) != frozen["files"]:
        raise ValueError("complete frozen production9f source manifest changed")
    if shutil.disk_usage(Path(args.output).resolve().parent).free < protocol["resources"]["initial_free_bytes_minimum"]:
        raise ValueError("insufficient fixed output reservation")
    return {"source_files": contract(), "root_go_sha256": sha(args.root_go),
            "root_go": go, "production_source_sha256": frozen["source_sha256"]}


def limits(protocol):
    runtime, caps = protocol["runtime"], protocol["resources"]
    if platform.system() != runtime["platform"] or platform.python_version() != runtime["python"]:
        raise ValueError("fixed Linux/Python runtime differs")
    for name in ("numpy", "scipy"):
        if importlib.metadata.version(name) != runtime[name]:
            raise ValueError("fixed numerical package version differs")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # Kernel-enforced limits apply only after plan/admission persistence. Default
    # SIGALRM preserves the pre-call ledger if a native call never returns.
    resource.setrlimit(resource.RLIMIT_AS, (caps["address_space_bytes"], caps["address_space_bytes"]))
    resource.setrlimit(resource.RLIMIT_CPU, (caps["cpu_limit_seconds"], caps["cpu_limit_seconds"]))
    signal.alarm(caps["wall_alarm_seconds"])


class Gate:
    """Every admitted public numerical dispatch has a flushed attempt prefix."""
    def __init__(self, output, protocol):
        self.output, self.protocol = output, protocol
        self.calls, self.owners = [], {}
        self.started = time.perf_counter()
        self.context = {}

    def event(self, kind, **value):
        row = {"kind": kind, "elapsed_seconds": time.perf_counter()-self.started, **value}
        with (self.output/"events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def snapshot(self, label, supplier):
        try:
            value = json.loads(json.dumps(supplier(), allow_nan=False))
            self.owners[label] = {"actual_last_snapshot": value, "unknown_after_prefix": False}
        except BaseException as exc:  # noqa: BLE001 -- preserve unknown counter tails
            previous=self.owners.get(label,{"actual_last_snapshot":None})
            self.owners[label]={"actual_last_snapshot":previous["actual_last_snapshot"],
                "unknown_after_prefix":True,"snapshot_failure_type":type(exc).__name__}
        return self.owners[label]

    def run(self, label, function, supplier):
        if len(self.calls) >= self.protocol["resources"]["maximum_recorded_public_dispatches"]:
            raise ValueError("fixed dispatch cap exceeded before numerical call")
        if time.perf_counter()-self.started >= self.protocol["resources"]["wall_alarm_seconds"]:
            raise ValueError("fixed wall cap exceeded before numerical call")
        row = {"index": len(self.calls), "label": label, "status": "attempted",
               "before": self.snapshot(label, supplier)}
        self.calls.append(row)
        self.event("public_dispatch_attempt", **row)
        try:
            result = function()
        except BaseException as exc:
            row.update(status="raised", exception_type=type(exc).__name__,
                       after=self.snapshot(label, supplier), unknown_native_tail=True)
            self.event("public_dispatch_raised", **row)
            raise
        row.update(status="returned", after=self.snapshot(label, supplier))
        self.event("public_dispatch_returned", **row)
        return result

    def evidence(self):
        # No numerical callback is queried here after a failure.
        return {"calls": self.calls, "last_observed_owner_prefixes": self.owners,
                "retained_context": self.context}


def imports(repo):
    """Only after plan persistence/GO/limits; verify loaded origins, not names."""
    package = (Path(repo)/"src/wavecert").resolve()
    sys.path.insert(0, str(package.parent))
    loaded = {}
    for name in ("cached_jacobian", "public_map_builder"):
        path = HERE/(name+".py")
        if name in sys.modules:
            raise ValueError("component module was preloaded before the isolated source gate")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    np = importlib.import_module("numpy")
    scipy = importlib.import_module("scipy")
    for module in (np, scipy):
        intended=Path(importlib.metadata.distribution(module.__name__).locate_file(module.__name__+"/__init__.py")).resolve()
        if Path(module.__file__).resolve()!=intended or module.__version__!=importlib.metadata.version(module.__name__):
            raise ValueError("numerical distribution/loaded module origin mismatch")
    helm = importlib.import_module("wavecert.physics.helmholtz")
    exact = importlib.import_module("wavecert.physics.exact_merit")
    for module, relative in ((helm, "physics/helmholtz.py"), (exact, "physics/exact_merit.py")):
        if Path(module.__file__).resolve() != package/relative:
            raise ValueError("foreign imported native module origin")
    for cls, names, path in ((helm.Helmholtz2D, ("operator", "source", "solve_state", "data_jvp", "data_vjp", "receiver_resolvent_matrix", "cost_snapshot"), package/"physics/helmholtz.py"),
            (exact.ExactMeritOracle, ("evaluate", "export_merit_fields", "validate_merit_fields", "cost_snapshot"), package/"physics/exact_merit.py")):
        for name in names:
            if Path(inspect.getsourcefile(getattr(cls, name))).resolve() != path:
                raise ValueError("foreign executing native public method")
    return np, helm, exact, loaded["public_map_builder"]


def numeric_worker(args, protocol, gate):
    np, helm, exact, binding = imports(args.repo)
    layout, physics_config = binding.current_full_layout(), dict(protocol["physics"])
    criteria=protocol["criteria_pending_independent_review"]
    physics_config["shape"] = tuple(physics_config["shape"])
    checks, payload, owners = [], {}, {}

    def require(value, label, **scalars):
        scalars={name:({"nonfinite_scalar":str(item)} if isinstance(item,float) and not math.isfinite(item) else item)
            for name,item in scalars.items()}
        checks.append({"check": label, "pass": bool(value), **scalars})
        gate.context["checks"] = checks
        if not value:
            raise ValueError("registered native integration check failed: "+label)

    def physics(label, cache=0):
        owner = helm.Helmholtz2D(**physics_config, factorization_cache_size=cache)
        require(type(owner) is helm.Helmholtz2D and not any(owner.cost_snapshot().values()), label+" actual cold public physics")
        owners[label] = owner
        return owner

    def compare(actual, expected, label, tolerance):
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), label+" finite complete shapes")
        denominator = float(np.linalg.norm(expected))
        require(denominator > protocol["criteria_pending_independent_review"]["nonzero_norm_floor"], label+" informative norm", norm=denominator)
        error = float(np.linalg.norm(actual-expected)/denominator)
        require(error <= tolerance, label+" relative error", relative_error=error, tolerance=tolerance)

    def vector(data):
        # All global blocks then ordered receiver slots; never deduplicate data.
        value = np.asarray(data, dtype=np.complex128).reshape(-1)
        return np.concatenate((value.real, value.imag))

    def forward(label, model, source_multipliers=None):
        owner = physics(label)
        data, states = [], []
        for frequency, sources, _ in layout.frames:
            rhs = np.column_stack([owner.source(source, amplitude=1. if source_multipliers is None else source_multipliers[j]) for j, source in enumerate(sources)])
            state = gate.run(label, lambda f=frequency, q=rhs: owner.solve_state(model, q, f), owner.cost_snapshot)
            require(state.shape == (layout.n, len(sources)) and np.isfinite(state).all(), label+" finite source fields")
            data.extend(state[list(layout.receivers)].T)
            states.append(state)
        return np.asarray(data), states

    def counts(actual, expected, label):
        require(all(actual.get(key, 0) == value for key, value in expected.items()), label+" actual public counts", actual=actual, expected=expected)

    m0 = np.full(layout.n, 1/protocol["initial_velocity"]**2, dtype=np.float64)
    mobs = np.full(layout.n, 1/protocol["observation_velocity"]**2, dtype=np.float64)
    i, j = np.indices(physics_config["shape"])
    p = (m0.reshape(physics_config["shape"])*(1+.2*np.sin(2*np.pi*i/48)*np.cos(2*np.pi*j/48))/1.2).reshape(-1)
    k = np.arange(2*layout.complex_terms, dtype=np.float64)+1
    z = np.sin(.017*k)+np.cos(.013*k)
    z /= np.linalg.norm(z)
    observations, _ = forward("observation_generation", mobs)
    counts(owners["observation_generation"].cost_snapshot(), {"factorizations":8,"forward_rhs":48,"stability_rhs":0}, "observations")
    gate.event("all_synthetic_observations_complete_before_policy", data_sha256=hashlib.sha256(observations.tobytes()).hexdigest())
    blocks = tuple(helm.SurveyBlock(source, frequency, observations[index].copy(), f"{source}@{frequency}")
        for frequency, sources, indices in layout.frames for source, index in zip(sources, indices, strict=True))
    template = physics("policy_template")
    native = exact.ExactMeritOracle(template, blocks, layout.receivers)
    require(type(native) is exact.ExactMeritOracle, "actual native receipt issuer")
    callbacks = binding.frozen_native_binding(native, reference_package_root=Path(args.repo)/"src/wavecert", initial_model=m0, layout=layout)
    gate.context["component_last_snapshot"] = callbacks.cost_snapshot()
    try:
        f0 = gate.run("policy", lambda: callbacks.fun(m0), callbacks.cost_snapshot)
        require(f0.shape == (1920,), "all1920 normalized residual terms")
        initial = callbacks.latest_fun
        require(type(initial.native_receipt) is exact.ExactMeritReceipt, "genuine native receipt type")
        ratio, norm_cost = initial.native_receipt.objective/callbacks.phi0, .5*float(f0@f0)
        require(math.isclose(norm_cost, ratio, rel_tol=criteria["normalized_objective_relative_error"], abs_tol=criteria["normalized_objective_absolute_error"]), "normalized residual cost equals nativePhi/Phi0", residual_cost=norm_cost, native_ratio=ratio)
        counts(native.cost_snapshot()["physics"], {"factorizations":8,"forward_rhs":48,"tangent_rhs":0,"adjoint_rhs":0,"stability_rhs":0}, "initial policy")
        operator = gate.run("policy", lambda: callbacks.jac(m0), callbacks.cost_snapshot)
        counts(native.cost_snapshot()["physics"], {"factorizations":8,"forward_rhs":48,"tangent_rhs":0,"adjoint_rhs":0,"stability_rhs":0}, "Jacobian did not re-solve native U")
        batches = callbacks.builder.cost_snapshot()["cold_owner_batches"]
        require(len(batches)==8 and all(row["actual_complete_batch_counts_verified"] and not row["unknown_work"] for row in batches), "all8 actual cold receiver-map batches")
        counts({key:sum(row["last_known_physics_prefix"].get(key,0) for row in batches) for key in ("factorizations","stability_rhs","forward_rhs")}, {"factorizations":8,"stability_rhs":112,"forward_rhs":0}, "incremental receiver maps no hidden U")
        cached_jp = gate.run("policy", lambda: operator@p, callbacks.cost_snapshot)
        cached_star = gate.run("policy", lambda: operator.T@z, callbacks.cost_snapshot)
        left, right = float(z@cached_jp), float(p@cached_star)
        denominator = max(abs(left), abs(right), float(np.linalg.norm(z)*np.linalg.norm(cached_jp)), 1e-12)
        require(abs(left-right)/denominator <= criteria["relative_real_adjoint_error"], "real adjoint identity", left=left, right=right)

        jvp_owner, vjp_owner = physics("public_jvp",8), physics("public_vjp",8)
        reference_jp, reference_star = [], np.zeros(layout.n, dtype=np.float64)
        complex_dual = (z[:layout.complex_terms]+1j*z[layout.complex_terms:]).reshape(layout.blocks, len(layout.receivers))/math.sqrt(callbacks.phi0)
        for frequency, sources, indices in layout.frames:
            for source, index in zip(sources, indices, strict=True):
                q = jvp_owner.source(source)
                reference_jp.append(gate.run("public_jvp",lambda f=frequency,q=q: jvp_owner.data_jvp(m0,q,layout.receivers,f,p), jvp_owner.cost_snapshot))
                q = vjp_owner.source(source)
                reference_star += gate.run("public_vjp",lambda f=frequency,q=q,index=index: vjp_owner.data_vjp(m0,q,layout.receivers,f,complex_dual[index]), vjp_owner.cost_snapshot)
        reference_jp = vector(reference_jp)/math.sqrt(callbacks.phi0)
        compare(cached_jp,reference_jp,"cached JVP versus public redo-U tangent",criteria["relative_vector_error"])
        compare(cached_star,reference_star,"cached Jstar versus public redo-U adjoint",criteria["relative_vector_error"])
        counts(jvp_owner.cost_snapshot(),{"factorizations":8,"forward_rhs":48,"tangent_rhs":48,"adjoint_rhs":0},"public JVP paid U")
        counts(vjp_owner.cost_snapshot(),{"factorizations":8,"forward_rhs":48,"adjoint_rhs":48,"tangent_rhs":0},"public VJP paid U")

        multipliers = tuple(complex(*pair) for pair in protocol["complex_source_multipliers"])
        _, complex_states = forward("complex_source_validation",m0,multipliers)
        map_owner = physics("full_receiver_map_validation")
        covered_rows = []
        for frame, complex_state in zip(initial.frames, complex_states, strict=True):
            rows = gate.run("full_receiver_map_validation",lambda f=frame.frequency,receivers=frame.unique: map_owner.receiver_resolvent_matrix(m0,f,receivers), map_owner.cost_snapshot)
            require(frame.symmetric, "actual native transpose symmetry measured")
            chosen = np.array([rows[frame.unique.index(source)] for source in frame.sources])
            compare((frame.state/(1/physics_config["spacing"]**2)).T,chosen,"native U.T/a covered rows",criteria["receiver_row_relative_error"])
            amplitudes = np.asarray(multipliers)/physics_config["spacing"]**2
            compare((complex_state/amplitudes).T,chosen,"complex U.T/a with no conjugation",criteria["receiver_row_relative_error"])
            covered_rows.append(chosen)
        counts(map_owner.cost_snapshot(),{"factorizations":8,"stability_rhs":160,"forward_rhs":0},"full reference map")

        phi_dd = float(f0@cached_jp)
        cosine = abs(phi_dd)/float(np.linalg.norm(f0)*np.linalg.norm(cached_jp))
        require(cosine >= criteria["nontrivial_relative_dd_cosine"] and abs(phi_dd)>criteria["nonzero_norm_floor"],"informative fixed objective direction",cosine=cosine,derivative=phi_dd)
        fd_vectors=[]
        for step in protocol["fd_steps"]:
            pair=[]
            for sign in (-1,1):
                model=m0+sign*step*p
                require(np.isfinite(model).all() and np.all(model>0),"fixed FD model finite positive")
                data,_=forward(f"fd_{step}_{sign}",model)
                pair.append(vector(data-observations)/math.sqrt(callbacks.phi0))
            derivative=(pair[1]-pair[0])/(2*step)
            compare(derivative,cached_jp,"fixed centered residual FD",criteria["fd_jvp_relative_error_each_step"])
            fd_phi=(.5*float(pair[1]@pair[1])-.5*float(pair[0]@pair[0]))/(2*step)
            error=abs(fd_phi-phi_dd)/max(abs(fd_phi),abs(phi_dd),1e-12)
            require(error<=criteria["fd_objective_relative_error_each_step"],"fixed centered nativePhi FD",epsilon=step,relative_error=error)
            fd_vectors.extend(pair)

        trial=m0+protocol["fd_steps"][0]*p
        gate.run("policy",lambda: callbacks.fun(trial),callbacks.cost_snapshot)
        repeat=gate.run("policy",lambda: operator@p,callbacks.cost_snapshot)
        require(np.array_equal(repeat,cached_jp),"old field operator fixed after new native trial")
        before=native.cost_snapshot()["physics"]
        try:
            gate.run("expected_model_rejection",lambda: callbacks.jac(m0+.002*p),callbacks.cost_snapshot)
        except ValueError:
            require(native.cost_snapshot()["physics"]==before,"unissued model query rejected before new physics work")
            gate.event("expected_source_bound_model_guard_rejection",physics_prefix_unchanged=True)
        else:
            raise ValueError("unissued model Jacobian was not rejected")
        foreign=exact.ExactMeritOracle(physics("foreign_template"),blocks,layout.receivers)
        try:
            gate.run("foreign_owner_validation",lambda: foreign.validate_merit_fields(initial.native_snapshot),foreign.cost_snapshot)
        except ValueError:
            require(not any(foreign.cost_snapshot()["physics"].values()),"foreign export rejected without physics solve")
            gate.event("expected_source_bound_owner_guard_rejection",physics_prefix_zero=True)
        else:
            raise ValueError("foreign issued snapshot accepted")

        payload.update(m0=m0,observation_model=mobs,observations=observations,direction=p,dual=z,initial_residual=f0,
            cached_jvp=cached_jp,reference_jvp=reference_jp,cached_vjp=cached_star,reference_vjp=reference_star,
            fd_residual_vectors=np.asarray(fd_vectors),native_U=np.asarray([frame.state for frame in initial.frames]),
            reference_C_covered=np.asarray(covered_rows),complex_U=np.asarray(complex_states))
        require(sum(value.nbytes for value in payload.values())<=protocol["resources"]["retained_array_payload_cap_bytes"],"retained finite array byte cap")
        require(all(np.isfinite(value).all() for value in payload.values()),"all retained arrays finite")
        np.savez(gate.output/"arrays.npz",**payload)
        final_callback=callbacks.cost_snapshot()
        physics_counts={name:owner.cost_snapshot() for name,owner in owners.items()}
        physics_counts["actual_native_policy"]=native.cost_snapshot()["physics"]
        physics_counts["actual_foreign_native_validation"]=foreign.cost_snapshot()["physics"]
        physics_counts["component_cold_map_batches"]={key:sum(row["last_known_physics_prefix"].get(key,0) for row in batches) for key in ("factorizations","forward_rhs","tangent_rhs","adjoint_rhs","stability_rhs")}
        total={key:sum(row.get(key,0) for row in physics_counts.values()) for key in protocol["expected_complete_disjoint_counts"]}
        counts(total,protocol["expected_complete_disjoint_counts"],"complete disjoint actual work")
        require(not final_callback["native"]["unknown_since_prefix"] and not final_callback["map_builder"]["unknown_work_present"],"completed numerical work known")
        return {"status":"NATIVE_FIELD_INTEGRATION_CHECKS_PASS_PENDING_EXTERNAL_WAIT", "checks":checks,
            "native_initial_header":initial.native_receipt.to_dict(),"component_costs":final_callback,
            "actual_disjoint_physics_counts":physics_counts,"total_actual_physics_counts":total,
            "array_metadata":{name:{"shape":list(value.shape),"dtype":str(value.dtype),"bytes":value.nbytes,
                "sha256":hashlib.sha256(value.tobytes()).hexdigest()} for name,value in payload.items()},
            "limits":"Shared frozen public sparse LU backend, not a fully independent solver, rigorous enclosure, recovery/physics/novelty result or student final design."}
    finally:
        # Component snapshot observes stored prefixes only; no native re-query.
        gate.context["component_last_snapshot"]=callbacks.cost_snapshot()


def lifecycle(args, protocol, admission, *, worker=numeric_worker, limiter=limits):
    output=Path(args.output).resolve()
    output.mkdir()  # fresh path; no retry or reuse
    gate=Gate(output,protocol)
    # Complete fixed plan and exact source/GO are durable BEFORE limiter/imports/PDE.
    write_new(output/"plan.json",protocol)
    write_new(output/"admission.json",admission)
    gate.event("full_plan_and_admission_persisted_before_numerical_imports")
    started=time.perf_counter()
    try:
        limiter(protocol)
        result=worker(args,protocol,gate)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        peak=int(usage.ru_maxrss)*(1024 if platform.system()=="Linux" else 1)
        if peak>protocol["resources"]["peak_RSS_review_bytes"]:
            raise ValueError("observed worker RSS exceeds fixed review cap")
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>protocol["resources"]["output_cap_bytes"]:
            raise ValueError("fixed output cap exceeded")
        result.update(source_files=contract(),observed_peak_RSS_bytes=peak,
            entry_through_pre_result_serialization_seconds=time.perf_counter()-started,
            actual_public_attempt_prefixes=gate.evidence(),
            output_artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            actual_external_wait_required=True,scientific_goals_complete=False)
        write_new(output/"result.json",result)
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>protocol["resources"]["output_cap_bytes"]:
            raise ValueError("fixed output cap exceeded including result serialization")
        return 0
    except BaseException as exc:  # noqa: BLE001 -- retain interrupt/serialization evidence and return explicit failure
        write_new(output/"failure.json",{"status":"FAILED_OR_INTERRUPTED_NO_RETRY","exception_type":type(exc).__name__,
            "message":str(exc),"source_files":contract(),"evidence":gate.evidence(),
            "native_tail_unknown_if_dispatch_did_not_return":True,"scientific_goals_complete":False})
        return 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--root-go",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    protocol=json.loads((HERE/"protocol.json").read_bytes())
    if not args.execute:
        print(json.dumps({"status":"HELD_NATIVE_INTEGRATION_SOURCE_ONLY","source_files":contract(),
            "native_execution_authorized":False,"numerical_imports":0,"PDE_calls":0}))
        return 0
    return lifecycle(args,protocol,admit(args,protocol))


if __name__=="__main__":
    raise SystemExit(main())
