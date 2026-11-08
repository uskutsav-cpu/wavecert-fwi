"""Exact integer geometry/count metadata only; no NumPy or project imports."""
import hashlib
import itertools
import json
from pathlib import Path

OUTER = Path(__file__).resolve().parents[3]
REPO = OUTER/"work/wavecert-fwi"
STAGE = Path(__file__).resolve().parent
PROTOCOL = "work/new-phase-staging/frequency_continuation_benchmark_02/protocol.json"
PINS = {
    "work/wavecert-fwi/src/wavecert/physics/helmholtz.py": "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131",
    "work/wavecert-fwi/src/wavecert/physics/exact_merit.py": "0fec1f215b28e36017c1e28409d7f7e909a03cbf5b717de5d23eb8d52eb26810",
    "work/wavecert-fwi/results/provisional_software_benchmark/matched_cost_pilot/verification.py": "b7bee25257caf456d52932962ceba2fde00197f0a78b585cf90c138e795bba4c",
    "work/new-phase-staging/frequency_continuation_benchmark_02/run_study.py": "a5f4a7bd38350ea1958160d0a5d24fb29b03e986b2601117c1bc7fd399394558",
    PROTOCOL: "5c2838eaa1142ae7497717d7a4dfc46c76df4fb5ff8b3e016d5479eb4b51af1e",
    "work/new-phase-staging/frequency_continuation_benchmark_02/original_protocol.json": "fb6b304fd719441be8cd6eb8d101ca864d0856254f40618583d8d986fd1c945c"
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    files = [REPO/"pyproject.toml", *REPO.glob("requirements*.txt"), *(REPO/"src").rglob("*.py"),
        *(p for p in (REPO/"configs").rglob("*") if p.is_file()),
        *(p for p in (REPO/"scripts").rglob("*") if p.suffix in {".py", ".sh"}),
        *(REPO/".github/workflows").glob("*.yml")]
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(path.relative_to(REPO).as_posix().encode()+b"\0"+sha(path).encode()+b"\0")
    return dict(files=len(set(files)), sha256=digest.hexdigest())


def refine(index):
    z, x = divmod(index, 24)
    return (2*z+1)*49+2*x+1


def counts(frequencies):
    raw_complex, duplicate_complex = len(frequencies)*6*20, len(frequencies)*15
    real_row_ceiling = 2*(raw_complex-duplicate_complex)
    return dict(frequencies_hz=frequencies, raw_complex_slots=raw_complex,
        raw_real_and_imaginary_slots=2*raw_complex, reciprocal_duplicate_complex_rows=duplicate_complex,
        potentially_distinct_complex_slots=raw_complex-duplicate_complex,
        real_Jacobian_rank_upper_bound=real_row_ceiling,
        unrestricted_real_parameter_count=2401,
        real_linearized_nullity_lower_bound=2401-real_row_ceiling)


def main():
    output = STAGE/"geometry_ceiling_metadata_01.json"
    assert not output.exists()
    assert all(sha(OUTER/name) == expected for name, expected in PINS.items())
    manifest_before = source_manifest()
    assert manifest_before == dict(files=161, sha256="9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da")
    protocol = json.loads((OUTER/PROTOCOL).read_bytes())
    sources = protocol["geometry"]["original_source_indices"]
    receivers = protocol["geometry"]["original_receiver_indices"]
    assert sources == [50, 54, 58, 61, 65, 69] and receivers == list(range(50, 70))
    mapped_sources, mapped_receivers = list(map(refine, sources)), list(map(refine, receivers))
    assert mapped_sources == [250, 258, 266, 272, 280, 288]
    assert len(set(mapped_receivers)) == 20 and set(mapped_sources) <= set(mapped_receivers)
    receiver_slots = [mapped_receivers.index(s) for s in mapped_sources]
    pairs = []
    for i, j in itertools.combinations(range(6), 2):
        pairs.append(dict(source_ordinals=[i, j], old_source_indices=[sources[i], sources[j]],
            native_source_indices=[mapped_sources[i], mapped_sources[j]],
            receiver_slots=[receiver_slots[j], receiver_slots[i]],
            duplicate_rows_within_one_frequency=[20*i+receiver_slots[j], 20*j+receiver_slots[i]]))
    assert len(pairs) == 15 and len({r for p in pairs for r in p["duplicate_rows_within_one_frequency"]}) == 30
    all8 = protocol["available_frequencies_hz"]
    original4 = protocol["original_frequencies_hz"]
    assert all8 == [.25, .5, 1., 2., 2.5, 3., 3.5, 4.] and original4 == [2.5, 3., 3.5, 4.]
    result = dict(status="SOURCE_METADATA_ONLY_PENDING_INDEPENDENT_MATH_REVIEW", source_pins=PINS,
        source_manifest_before_after=manifest_before, native_shape=[49, 49], native_m_units="s^2/km^2",
        mapping="old(z,x) -> (2z+1,2x+1), C-order49; source-major20receiver rows within each frequency",
        original_source_indices=sources, original_receiver_indices=receivers,
        native_source_indices=mapped_sources, native_receiver_indices=mapped_receivers,
        source_positions_in_receiver_order=receiver_slots, reciprocal_pairs=pairs,
        reciprocal_duplicate_rows_in_full8_vector=[dict(frequency_hz=f,
            row_pairs=[[120*k+a, 120*k+b] for a, b in (p["duplicate_rows_within_one_frequency"] for p in pairs)])
            for k, f in enumerate(all8)],
        all8_map=counts(all8), original4_submap=counts(original4),
        per_frequency_count_identity="6*(20-6)+6*(6+1)/2=84+21=105;120-15=105",
        operator="A(m,omega)=K-omega^2 diag(m)+i omega diag(d), K^T=K, d fixed real diagonal",
        source_and_receiver="q_s=e_s/h^2 with common nonzero amplitude; receiver e_r^T; h=.025km",
        reciprocity="A^T=A -> (A^-1)^T=A^-1 -> e_r^T A^-1 e_s=e_s^T A^-1 e_r; transpose, not conjugate transpose",
        derivative="DF_f(m)[dm]_(r,s)=(omega^2/h^2)*e_r^T A^-1 diag(dm) A^-1 e_s; swapped source/receiver rows remain equal",
        conditional_domain="At real positive m where every selected A(m,omega) is nonsingular, hence forward map is differentiable; ideal exact-arithmetic structural ceiling, not measured floating-point rank",
        numerical_rank_conditioning_or_actual_array_equality_computed=False,
        project_array_reads=0, PDE_or_optimizer_calls=0,
        discarded_or_reweighted_objective_terms=0, physical_acceptance=None,
        student_final_design_or_novelty_claim=False)
    assert source_manifest() == manifest_before
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(dict(status=result["status"], pair_count=len(pairs), all8=result["all8_map"], original4=result["original4_submap"]), sort_keys=True))


if __name__ == "__main__":
    main()
