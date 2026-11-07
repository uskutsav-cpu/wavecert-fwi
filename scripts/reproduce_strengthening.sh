#!/usr/bin/env bash
set -euo pipefail
wavecert_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$wavecert_root"
wavecert_python="${WAVECERT_PYTHON:-python3}"
export PYTHONPATH="$wavecert_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

"$wavecert_python" -m pytest -ra
"$wavecert_python" -m wavecert verify-physics --output results/independent_physics
"$wavecert_python" scripts/run_research_audit.py --output results/research_audit
"$wavecert_python" scripts/run_receiver_controls.py

if [[ "${1:-full}" == "quick" ]]; then
  wavecert_evidence_output=results/quick_evidence
  wavecert_evidence_models=1
  wavecert_evidence_iterations=2
  wavecert_evidence_scenarios=1
  "$wavecert_python" -m wavecert evidence-study --seeds 731 --cases-per-seed 1 \
    --iterations 2 --calibration-cases 10 --alpha 0.2 --scenarios clean \
    --output "$wavecert_evidence_output"
elif [[ "${1:-full}" == "full" ]]; then
  wavecert_evidence_output=results/evidence_study
  wavecert_evidence_models=60
  wavecert_evidence_iterations=12
  wavecert_evidence_scenarios=4
  "$wavecert_python" -m wavecert evidence-study --cases-per-seed 20 --iterations 12 \
    --output "$wavecert_evidence_output"
else
  echo "usage: scripts/reproduce_strengthening.sh [quick|full]" >&2
  exit 2
fi

"$wavecert_python" scripts/audit_evidence_study.py "$wavecert_evidence_output" \
  --repository "$wavecert_root" --expected-models "$wavecert_evidence_models" \
  --expected-iterations "$wavecert_evidence_iterations" \
  --expected-policies 9 --expected-scenarios "$wavecert_evidence_scenarios" \
  --output "${wavecert_evidence_output}_audit/summary.json"

"$wavecert_python" -m wavecert real-wavecert-study \
  --dataset data/reference/subsurfacegen_parent_split \
  --checkpoint checkpoints/fno_parent_grouped.pt --output results/grouped_wavecert
"$wavecert_python" -m wavecert real-wavecert-study \
  --dataset data/reference/subsurfacegen_parent_split \
  --checkpoint checkpoints/fno_parent_grouped_jvp.pt --output results/grouped_wavecert_jvp
"$wavecert_python" scripts/diagnose_development_generalization.py

echo "Core learned-model, independent-physics, and parent-held-out studies completed."
echo "Devito remains a separate optional baseline; see README."
