"""Plot only the completed fixed batching study; no model inference or timing."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.repository.resolve(), args.output.resolve()
    assert not out.exists() and not out.is_relative_to(root)
    out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(out / '.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study = root / 'results/batched_pod_development'
    summary = json.loads((study / 'summary.json').read_text())
    rows = list(csv.DictReader((study / 'cold_calls.csv').open(newline='')))
    assert summary['status'] == 'complete' and all(summary['checks'].values()) and len(rows) == 288
    methods = ['best_exact', 'raw_POD_scalar', 'raw_POD_batch', 'corrected_scalar96', 'corrected_batch96', 'corrected_batch64']
    labels = ['Exact\nobjective + gradient', 'Raw POD64\nscalar', 'Raw POD64\nbatched', 'Corrected POD64\nscalar, coarse96', 'Corrected POD64\nbatched, coarse96', 'Corrected POD64\nbatched, coarse64']
    identities = sorted({(r['parent_id'], r['slice_id'], int(r['case_index'])) for r in rows})
    assert len(identities) == 8 and len({i[0] for i in identities}) == 4
    fig, ax = plt.subplots(figsize=(13.6, 6.8))
    fig.subplots_adjust(left=.09, right=.985, bottom=.29, top=.78)
    colors = ['#2769a7', '#1b8b67', '#b17527', '#8a55a2']
    parents = sorted({i[0] for i in identities})
    numbers = {}
    for j, method in enumerate(methods):
        selected = [r for r in rows if r['method'] == method]
        assert len(selected) == 48
        medians = []
        for parent, slice_id, case_index in identities:
            group = [r for r in selected if int(r['case_index']) == case_index]
            assert len(group) == 6 and {int(r['repetition']) for r in group} == set(range(6))
            assert all(r['parent_id'] == parent and r['slice_id'] == slice_id for r in group)
            values = [1000 * float(r['total_cold_seconds']) for r in group]
            assert all(v > 0 for v in values)
            medians.append(float(np.median(values)))
        score = summary['descriptive_methods'][method]['equal_parent_mean_slice_median_total_cold_seconds'] * 1000
        # Each of the four parents has two slices; show exactly the declared score.
        parent_means = [np.mean([medians[k] for k, identity in enumerate(identities) if identity[0] == p]) for p in parents]
        assert np.isclose(score, np.mean(parent_means), rtol=1e-12)
        for k, (identity, value) in enumerate(zip(identities, medians, strict=True)):
            ax.scatter(j + (k-3.5)*.025, value, color=colors[parents.index(identity[0])], s=32, alpha=.85, zorder=3)
        ax.plot([j-.22, j+.22], [score, score], color='#202020', lw=2.4, zorder=4)
        first = [r for r in selected if r['repetition'] == '0']
        certified = sum(r['certified_descent'] == 'True' for r in first) if method.startswith('corrected') else None
        numbers[method] = dict(equal_parent_mean_slice_median_total_cold_ms=score, certified_slices=certified, n_slices=8)
        ax.text(j, max(medians)*1.14, f'{score:.2f} ms', ha='center', fontsize=11.5, weight='bold')
        if certified is not None:
            ax.text(j, max(medians)*1.42, f'{certified}/8 certified', ha='center', fontsize=10, color='#a14724' if certified < 8 else '#444444')
    ax.set_yscale('log')
    ax.set_ylim(3.4, 125)
    ax.set_xlim(-.5, 5.5)
    ax.set_xticks(range(6), labels, fontsize=10)
    ax.set_ylabel('Total cold setup + query time (milliseconds)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e6e6')
    fig.suptitle('Batching reduces dispatch cost; the certificate remains slower than exact', fontsize=17, weight='bold', y=.97)
    fig.text(.5, .87, 'Eight reused validation slices / four parents · point = median of six cold visits · black line = equal-parent score', ha='center', fontsize=11.5)
    fig.text(.09, .16, 'Different tasks: exact objective/gradient; raw approximate objective/gradient; corrected physical proposal plus bound.\n'
             'Rank64 retains all cases but is ineligible: 0/8 certified. Rank96 is a development recommendation only.\n'
             'All setup/current-model work paid; no equal-accuracy, inversion, field-scale or new-confirmation claim.', fontsize=10.5, color='#555555', linespacing=1.55, va='top')
    caption = ('Completed post-confirmation engineering study: all 288 balanced cold visits, eight existing validation slices/four parents, six fixed controls. '
               'Dots are per-slice medians of six visits and black bars are the fixed equal-parent mean of slice medians, not confidence intervals. '
               'All corrected96 cases certify; corrected64 certifies none despite finite available covering bounds. Uncertified raw controls were not assigned a certificate. '
               'The information tasks differ and ordinary-floating-point certificate limitations remain. No held-out rerun, automatic deployment, matched-accuracy or inversion speedup claim.')
    hashes = {}
    for extension in ['png', 'svg']:
        path = out / f'batched_development_costs.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'batched_development_costs.caption.txt').write_text(caption+'\n')
    inputs = {name:hashlib.sha256((study/name).read_bytes()).hexdigest() for name in ['summary.json', 'cold_calls.csv', 'protocol.json']}
    (out / 'numeric_and_input_hash_manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=inputs, outputs=hashes, values=numbers, caption=caption), indent=2)+'\n')
    print(json.dumps(numbers, indent=2))


if __name__ == '__main__':
    main()
