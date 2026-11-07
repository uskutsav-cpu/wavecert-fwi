"""PRIVATE fixed TRAIN/VAL DST defect-repair feasibility; no timings/public edits.

Classical deflated/inexact linear repair under the EXISTING coarse residual
certificate. No new theorem, convergence guarantee, NN, optimizer or heldout
study is claimed. Run --prepare first to pin every identity before outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndi

from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate

ROOT = Path('/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi')
WORK = Path(__file__).resolve().parent
PROTOCOL = WORK / 'fft_defect_feasibility_protocol.json'
OUTPUT = WORK / 'fft_defect_feasibility.json'
ORIGINAL = WORK / 'coarse_bound_feasibility.json'
BASIS = ROOT / 'results/pod_development/bases/state_tangent_rank64.npz'
BASIS_SHA = 'f2ae08cbcb251802ba7be92d69e87c28f425c5b1dfeede02fd94d6720c484447'
RANKS = (64, 96)
FREQUENCIES = (3., 4.)
CHECKPOINTS = (0, 1, 2, 4)
INITIALIZATIONS = ('pure_low_galerkin', 'pod64_old_coarse_corrected')


def dump_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def inputs():
    original = json.loads(ORIGINAL.read_text())
    selections = original['selected_models']
    assert len(selections) == 4 and {s['role'] for s in selections} == {'train', 'validation'}
    paths = [ORIGINAL, BASIS]
    data = ROOT / 'data/reference/subsurfacegen_parent_split'
    paths += [data / name for name in ('dataset_manifest.json', 'split_manifest.csv')]
    paths += [data / f'{role}{suffix}' for role in ('train', 'validation') for suffix in ('.npz', '_metadata.csv')]
    return selections, {str(path): sha256_file(path) for path in paths}


def prepare():
    selections, hashes = inputs()
    assert hashes[str(BASIS)] == BASIS_SHA
    assert Path(__import__('wavecert.physics.coarse_stability', fromlist=['x']).__file__).resolve() == ROOT / 'src/wavecert/physics/coarse_stability.py'
    protocol = {'schema_version': 1, 'created_utc_before_outcomes': datetime.now(timezone.utc).isoformat(),
        'scope': 'PRIVATE scalar mathematical feasibility; TRAIN/VAL only, no timings/heldout/test arrays/public code/NN fitting/optimizer.',
        'selected_models': selections, 'ranks': list(RANKS), 'frequencies_hz': list(FREQUENCIES),
        'checkpoints': list(CHECKPOINTS), 'initializations': list(INITIALIZATIONS),
        'geometry': {'shape': [24,24], 'spacing': .05, 'damping_width': 3, 'damping_strength': 2., 'source_index': 54, 'receiver_count': 20},
        'evaluation_model': 'Same cached raw m; smooth velocity with sigma1.6/reflect, then invert square. Observations from offline exact raw-model solve.',
        'high_inverse': 'DST-I orthonormal diagonal H: mask all fixed low modes, divide high modes by lambda_grid-omega^2*m_mid+i*omega*d_mid; low denominators1 then coefficients0. Adjoint conjugates denominator.',
        'midpoints': 'm_mid=(min(current smoothed m)+max(current smoothed m))/2; d_mid=(min(damping)+max(damping))/2. gamma>0 required.',
        'preconditioner': 'z=H(rhs); then z=ctx.correct(z,rhs,adjoint=...)[0], so projected residual is zero.',
        'sweep': 'u+=M^-1(q-Au), then lambda+=M_H^-1(P^T(Pu-d)-A^H lambda) using UPDATED u each sweep. No adaptive stopping/best checkpoint.',
        'k0': 'Pure low Galerkin state/adjoint, or frozenPOD64 state/adjoint with exactly the original sequential coarse correction.',
        'certificate': 'UNCHANGED oriented residual state bound + receiver-oriented adjoint bound, identical physical-gradient directional three-term inequality. No new theorem/convergence claim; ordinary floating point.',
        'outcomes': 'Retain all128 fixed rows (8 source/frequency cases per init/rank/checkpoint), including unavailability/numeric failure; preserve negative results.',
        'verification': 'Fine states/adjoints and tangent DD are OFFLINE checks only; residual/component/receiver/directional coverage and gradient quality. No online exact recovery.',
        'input_sha256': hashes, 'script_sha256': sha256_file(Path(__file__)), 'provenance': environment_record(ROOT),
        'cost_scope': 'Untimed feasibility. Contexts/POD raw fields reused across variants; counters include diagnostic proof checks and do not define independent runtime or a speedup.',
        'selection_scope': 'The original four cases were fixed by earlier train/validation metadata, not selected using these outcomes. Both ranks and all k retained; no IID or field-scale inference.'}
    dump_new(PROTOCOL, protocol)
    print(json.dumps({'status': 'protocol pinned BEFORE outcomes', 'path': str(PROTOCOL), 'sha256': sha256_file(PROTOCOL)}), flush=True)


class HighComplementRepair:
    def __init__(self, context, m, damping):
        self.context = context
        self.shape = tuple(context.physics_config['shape'])
        nz, nx = self.shape
        z, x = np.arange(1,nz+1)[:,None], np.arange(1,nx+1)[None,:]
        h = context.physics_config['spacing']
        eigen = 4/h**2*(np.sin(np.pi*z/(2*(nz+1)))**2+np.sin(np.pi*x/(2*(nx+1)))**2)
        self.mask = np.ones(self.shape, dtype=bool)
        modes = context._engine.mode_indices
        self.mask[modes[:,0]-1, modes[:,1]-1] = False
        midpoint_m = (float(np.min(m))+float(np.max(m)))/2
        midpoint_d = (float(np.min(damping))+float(np.max(damping)))/2
        self.denominator = eigen-context.omega**2*midpoint_m+1j*context.omega*midpoint_d
        assert context.available and context.gamma > 0
        assert float(np.min(self.denominator.real[self.mask])) >= context.gamma-1e-10
        self.denominator[~self.mask] = 1
        self.counts = {'high_dst_calls':0, 'high_idst_calls':0, 'high_dst_real_equivalents':0,
                       'high_idst_real_equivalents':0, 'high_coarse_corrected_inverse_applications':0}
        self.midpoints = {'m_mid': midpoint_m, 'd_mid': midpoint_d, 'minimum_high_real_denominator': float(np.min(self.denominator.real[self.mask]))}

    def apply(self, rhs, *, adjoint=False):
        coefficients = fft.dstn(rhs.reshape(self.shape), type=1, norm='ortho', workers=1)
        self.counts['high_dst_calls'] += 1
        self.counts['high_dst_real_equivalents'] += 2 if np.iscomplexobj(rhs) else 1
        coefficients = coefficients / (self.denominator.conjugate() if adjoint else self.denominator)
        coefficients[~self.mask] = 0
        value = fft.idstn(coefficients, type=1, norm='ortho', workers=1).reshape(-1)
        self.counts['high_idst_calls'] += 1
        self.counts['high_idst_real_equivalents'] += 2 if np.iscomplexobj(coefficients) else 1
        value, residual = self.context.correct(value, rhs, adjoint=adjoint)
        self.counts['high_coarse_corrected_inverse_applications'] += 1
        coarse = la.norm(self.context.project(residual))
        assert coarse <= 1e-9*max(1., float(la.norm(rhs)))
        return value


def inspect_fields(context, physics, m, q, observed, receivers, frequency, u, adjoint, exact, identity):
    exact_value, exact_gradient, exact_u, exact_adjoint = exact
    omega2 = context.omega**2
    residual = u[list(receivers)]-observed
    rp = context.apply_operator(u)-q
    ra = context.apply_operator(adjoint, adjoint=True)-physics.inject_receivers(residual, receivers)
    gradient = omega2*np.real(np.conjugate(adjoint)*u)
    norm = float(la.norm(gradient))
    assert np.all(np.isfinite(gradient)) and np.isfinite(norm)
    direction = -gradient/norm if norm > 1e-14 else np.zeros_like(gradient)
    approximate_dd, exact_dd = float(np.dot(gradient,direction)), float(np.dot(exact_gradient,direction))
    tangent = physics.solve_tangent(m,exact_u,direction,frequency)
    tangent_dd = float(np.vdot(exact_u[list(receivers)]-observed,tangent[list(receivers)]).real)
    eta_u = context.oriented_error_bounds(rp)
    receiver = context.receiver_constants(receivers)
    eta_receiver = float(np.dot([receiver['pB'],receiver['pQ']],eta_u))
    eta_adjoint = context.oriented_adjoint_error_bounds(ra,eta_receiver,receivers)
    actual_u = context.component_norms(exact_u-u)
    actual_adjoint = context.component_norms(exact_adjoint-adjoint)
    actual_receiver = float(la.norm((exact_u-u)[list(receivers)]))
    first = float(context.component_norms(direction*u) @ eta_adjoint)
    second = float(context.component_norms(direction*adjoint) @ eta_u)
    cross = float(la.norm(direction,np.inf)*la.norm(eta_u)*la.norm(eta_adjoint))
    bound = omega2*(first+second+cross)
    error = abs(exact_dd-approximate_dd)
    gradient_norm = float(la.norm(exact_gradient))
    return {**identity, 'evaluation_complete': True, 'stability_available': True, 'reason': None,
        'exact_objective': exact_value, 'approximate_objective': .5*float(np.vdot(residual,residual).real),
        'exact_gradient_norm': gradient_norm, 'approximate_gradient_norm': norm,
        'gradient_cosine': float(np.clip(np.dot(exact_gradient,gradient)/gradient_norm/norm,-1,1)) if min(gradient_norm,norm)>1e-14 else 0.,
        'gradient_relative_l2': float(la.norm(gradient-exact_gradient)/max(gradient_norm,1e-15)),
        'direction_norm': float(la.norm(direction)), 'exact_directional_derivative': exact_dd,
        'tangent_directional_derivative': tangent_dd, 'tangent_dd_agreement': abs(tangent_dd-exact_dd) <= 1e-10*max(1.,abs(exact_dd)),
        'approximate_directional_derivative': approximate_dd, 'actual_directional_error': error,
        'directional_error_bound': float(bound), 'certified_upper_derivative': float(approximate_dd+bound),
        'true_descent': bool(exact_dd < 0), 'certified_descent': bool(approximate_dd+bound < 0 and norm>1e-14),
        'exact_fallback_required': bool(not (approximate_dd+bound < 0 and norm>1e-14)),
        'covers_directional_error': bool(error <= bound+1e-12),
        'primal_residual_components': context.component_norms(rp).tolist(),
        'adjoint_residual_components': context.component_norms(ra).tolist(),
        'eta_u_components': eta_u.tolist(), 'actual_u_error_components': actual_u.tolist(),
        'eta_adjoint_components': eta_adjoint.tolist(), 'actual_adjoint_error_components': actual_adjoint.tolist(),
        'covers_u_components': bool(np.all(actual_u <= eta_u+1e-9*np.maximum(1,actual_u))),
        'covers_adjoint_components': bool(np.all(actual_adjoint <= eta_adjoint+1e-9*np.maximum(1,actual_adjoint))),
        'eta_receiver_state': eta_receiver, 'actual_receiver_state_error': actual_receiver,
        'covers_receiver_state': bool(actual_receiver <= eta_receiver+1e-9*max(1.,actual_receiver)),
        'directional_bound_terms': [omega2*first,omega2*second,omega2*cross],
        'bound_to_descent_margin': float(bound/max(-approximate_dd,1e-15))}


def execute():
    assert not OUTPUT.exists(), 'preserve prior private outcomes'
    protocol = json.loads(PROTOCOL.read_text())
    selections, live_hashes = inputs()
    assert protocol['script_sha256'] == sha256_file(Path(__file__))
    assert live_hashes == protocol['input_sha256'] and selections == protocol['selected_models']
    assert source_digest(ROOT) == protocol['provenance']['source_sha256']
    physics = Helmholtz2D(shape=(24,24),spacing=.05,damping_width=3,damping_strength=2.)
    truth = Helmholtz2D(shape=(24,24),spacing=.05,damping_width=3,damping_strength=2.)
    geometry = SurveyGeometry.from_grid((24,24),n_sources=6,n_receivers=20,source_depth=2,receiver_depth=2,margin=2)
    q, receivers = physics.source(54), tuple(geometry.receiver_indices)
    assert geometry.source_indices[1] == 54 and len(receivers)==20
    pod = PODGalerkinSurrogate.load(BASIS,rank=64)
    rows, contexts = [], []
    for selection in selections:
        role = selection['role']
        assert role in ('train','validation')
        dataset = ROOT/'data/reference/subsurfacegen_parent_split'
        with (dataset/f'{role}_metadata.csv').open(newline='') as stream:
            metadata = list(csv.DictReader(stream))
        record = metadata[selection['array_index']]
        assert all(record[key] == selection[key] for key in ('model_id','slice_id','model_type'))
        with np.load(dataset/f'{role}.npz',allow_pickle=False) as archive:
            raw = archive['models'][selection['array_index']].copy()
        assert digest(raw) == selection['raw_m_sha256']
        m = (1/ndi.gaussian_filter(1/np.sqrt(raw.astype(float)),sigma=1.6,mode='reflect')**2).reshape(-1)
        for frequency in FREQUENCIES:
            observed = truth.solve_state(raw.astype(float).reshape(-1),q,frequency)[list(receivers)]
            exact = physics.objective_and_gradient(m,q,observed,receivers,frequency)
            _, _, pod_u, pod_adjoint = pod.objective_and_gradient(m,q,observed,receivers,frequency)
            for rank in RANKS:
                engine = FastCoarseStability(physics,rank=rank)
                context = engine.evaluate(m,frequency)
                contexts.append({**selection,'frequency_hz':frequency,'rank':rank,**context.to_dict()})
                for initialization in INITIALIZATIONS:
                    identity = {**selection,'frequency_hz':frequency,'rank':rank,'initialization':initialization}
                    failure = None
                    if context.available:
                        repair = HighComplementRepair(context,m,physics.damping)
                        if initialization == 'pure_low_galerkin':
                            u = context.lift(context.solve_coarse(context.project(q)))
                            adjoint_rhs = physics.inject_receivers(u[list(receivers)]-observed,receivers)
                            adjoint = context.lift(context.solve_coarse(context.project(adjoint_rhs),adjoint=True))
                        else:
                            u, _ = context.correct(pod_u,q)
                            adjoint_rhs = physics.inject_receivers(u[list(receivers)]-observed,receivers)
                            adjoint, _ = context.correct(pod_adjoint,adjoint_rhs,adjoint=True)
                    for step in range(5):
                        if context.available and failure is None and step:
                            try:
                                u += repair.apply(q-context.apply_operator(u))
                                rhs = physics.inject_receivers(u[list(receivers)]-observed,receivers)
                                adjoint += repair.apply(rhs-context.apply_operator(adjoint,adjoint=True),adjoint=True)
                            except (ValueError,FloatingPointError,AssertionError) as error:
                                failure = f'{type(error).__name__}: {error}'
                        if step not in CHECKPOINTS:
                            continue
                        row = {**identity,'repair_sweeps':step,'evaluation_complete':False,
                            'stability_available':bool(context.available),'reason':failure or context.reason,
                            'certified_descent':False,'exact_fallback_required':True,
                            'directional_error_bound':None,'certified_upper_derivative':None,'covers_directional_error':None}
                        if context.available and failure is None:
                            try:
                                row = inspect_fields(context,physics,m,q,observed,receivers,frequency,u,adjoint,exact,{**identity,'repair_sweeps':step})
                                row['high_complement'] = {**repair.midpoints,**repair.counts}
                                row['paid_context_counters_cumulative'] = context.cost_snapshot()
                            except (ValueError,FloatingPointError,AssertionError) as error:
                                failure = f'{type(error).__name__}: {error}'
                                row.update(reason=failure)
                        rows.append(row)
        print(json.dumps({'completed_model':selection['model_id'],'role':role,'saved_rows':len(rows)}),flush=True)
    summary = {}
    for initialization in INITIALIZATIONS:
        summary[initialization] = {}
        for rank in RANKS:
            summary[initialization][str(rank)] = {}
            for step in CHECKPOINTS:
                group = [r for r in rows if r['initialization']==initialization and r['rank']==rank and r['repair_sweeps']==step]
                valid = [r for r in group if r['evaluation_complete']]
                summary[initialization][str(rank)][str(step)] = {'cases':len(group),'complete':len(valid),
                    'certified_descent':sum(r['certified_descent'] for r in group),
                    'true_descent':sum(r['true_descent'] for r in valid),
                    'false_certifications':sum(r['certified_descent'] and not r['true_descent'] for r in valid),
                    'all_coverage_checks_pass':all(all(r[key] for key in ('covers_directional_error','covers_u_components','covers_adjoint_components','covers_receiver_state','tangent_dd_agreement')) for r in valid),
                    'mean_gradient_cosine':float(np.mean([r['gradient_cosine'] for r in valid])) if valid else None,
                    'maximum_gradient_relative_l2':max((r['gradient_relative_l2'] for r in valid),default=None),
                    'maximum_bound_to_descent_margin':max((r['bound_to_descent_margin'] for r in valid),default=None),
                    'median_bound_to_descent_margin':float(np.median([r['bound_to_descent_margin'] for r in valid])) if valid else None}
    assert len(rows)==128 and all(cell['cases']==8 for ranks in summary.values() for steps in ranks.values() for cell in steps.values())
    unchanged = source_digest(ROOT)==protocol['provenance']['source_sha256'] and inputs()[1]==live_hashes and sha256_file(Path(__file__))==protocol['script_sha256']
    report = {'schema_version':1,'study_status':'complete' if unchanged else 'invalid_source_or_input_change',
        'scope':protocol['scope'],'protocol_sha256':sha256_file(PROTOCOL),'script_sha256':sha256_file(Path(__file__)),
        'source_and_inputs_unchanged':unchanged,'main_repository_files_modified_by_this_script':False,
        'summary':summary,'cases':rows,'stability_contexts':contexts,'provenance_before':protocol['provenance'],
        'provenance_after':environment_record(ROOT),'offline_fine_verification_cost':physics.cost_snapshot(),
        'offline_truth_cost':truth.cost_snapshot(),'reused_pod_initial_field_cost':pod.cost_snapshot(),
        'limits':['Only four preselected TRAIN/VAL models, one source and two frequencies; no heldout or aggregate24 survey result.',
                  'Classical fixed inexact/deflated repair; existing certificate only, no novelty or convergence claim.',
                  'No timings or acceleration inference; reused contexts/initial fields/counters include private diagnostics.',
                  'Both ranks, both initializations and every predeclared checkpoint preserved; no adaptive/tuned best.']}
    dump_new(OUTPUT,report)
    print(json.dumps({'status':report['study_status'],'output':str(OUTPUT),'summary':summary}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    assert args.prepare != args.execute
    prepare() if args.prepare else execute()
