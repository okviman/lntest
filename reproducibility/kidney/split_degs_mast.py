"""MAST on kidney spot splitting with DEGs, the experiment of Fig 1g, h, on the first SPLITS replicates of each p_split.

The spot_split run with DEGs (--seed 0) draws its replicate seeds as the run without DEGs does. Replicate r at each
p_split is therefore split r of split_null_per_split.csv with DEGs planted. Each replicate is rebuilt from that seed by
spot_split.create_groups_from_shape_ids and spot_split.plant_degs (q and lfc as recorded in split_degs_results.csv),
after the capsule filter. It is scored by spot_split.de_test_single with MAST alone, through Seurat's FindMarkers with
its gene filters off (de_utils.run_mast_two_groups), with Bonferroni. MAST takes minutes per fit, so it has SPLITS
replicates where the other tests have 100.
The metrics of each replicate go to split_degs_mast_per_split.csv. Its precision at spot_split.RECALL_GRID goes to
split_degs_mast_curves.csv. Both are written batch by batch. A rerun keeps the replicates already in both.
spot_split.summarise_de summarises them into split_degs_mast_results.csv and split_degs_mast_pr_curve_data.json, in the
formats of spot_split. The figures show the TPR, FPR and PR curves of all four tests, the other three from the results
of spot_split with DEGs.

    python -m kidney.split_degs_mast              # from reproducibility/; about 25 minutes; needs R_HOME; resumes
    python -m kidney.split_degs_mast --plot-only  # redraw from the CSVs
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # before numpy is imported; the workers inherit it

import argparse
import json
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from kidney.fpr_plots import SPLIT
from kidney.split_null_mast import PER_SPLIT as NULL_SPLITS, capsules
from paths import fig_name, require_input, results_dir

SPLITS = 3       # the first replicates of each p_split
WORKERS = 4      # MAST fits at a time, one thread each
OUT = results_dir(__file__)
DEGS = OUT / 'split_degs_results.csv'
DEG_CURVES = OUT / 'split_degs_pr_curve_data.json'
PER_SPLIT = OUT / 'split_degs_mast_per_split.csv'
CURVES = OUT / 'split_degs_mast_curves.csv'
RESULTS = OUT / 'split_degs_mast_results.csv'
CURVE_JSON = OUT / 'split_degs_mast_pr_curve_data.json'
METRICS = ['n_total', 'n_selected', 'tp', 'fp', 'fn', 'tn', 'tpr', 'fpr', 'fnr', 'tnr', 'accuracy', 'precision',
           'pr_auc_curve', 'ap', 'pr_auc']
COLUMNS = ['p', 'split', 'seed'] + METRICS


def replicate(caps, p, seed, q, lfc):
    """One replicate's counts with its DEGs, as spot_split.run_single_iteration_de builds them."""
    from kidney.spot_split import create_groups_from_shape_ids, plant_degs
    np.random.seed(seed)
    X, Y = create_groups_from_shape_ids(*caps, p, seed=seed)
    return plant_degs(X, Y, q, lfc)


def mast_replicate(task):
    """One replicate's MAST metrics, as a per-split row, and its precision at RECALL_GRID."""
    from kidney.spot_split import de_test_single, interpolate_precision
    p, split, seed, X, Y, selected, signs = task
    m = de_test_single(X, Y, selected, signs, methods=('MAST',))['MAST']
    print(f"  p_split {p:.3f}, replicate {split}: TPR {m['tpr']:.3f}, FPR {m['fpr']:.4f}", flush=True)
    row = dict(zip(COLUMNS, [p, split, seed] + [m[k] for k in METRICS]))
    return row, interpolate_precision(m['precision_curve'], m['recall_curve'])


def run(seeds, q, lfc):
    """The replicates built here, WORKERS at a time, so that only their counts reach the workers; each batch written to
    PER_SPLIT and CURVES, and replicates already in both kept."""
    from kidney.spot_split import RECALL_GRID
    done = pd.read_csv(PER_SPLIT, float_precision="round_trip") if PER_SPLIT.exists() else pd.DataFrame(columns=COLUMNS)
    curves = (pd.read_csv(CURVES, float_precision="round_trip") if CURVES.exists()
              else pd.DataFrame(columns=['p', 'split', 'recall', 'precision']))
    have = (set(zip(done['p'].astype(float).round(4), done['split']))
            & set(zip(curves['p'].astype(float).round(4), curves['split'])))
    tasks = [(r.p, int(r.split), int(r.seed)) for r in seeds.itertuples() if (round(r.p, 4), r.split) not in have]
    caps = capsules() if tasks else None
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        for start in range(0, len(tasks), WORKERS):
            results = pool.map(mast_replicate, [(p, split, seed, *replicate(caps, p, seed, q, lfc))
                                                for p, split, seed in tasks[start:start + WORKERS]])
            done = pd.concat([done, pd.DataFrame([row for row, _ in results], columns=COLUMNS)], ignore_index=True)
            curves = pd.concat([curves] + [pd.DataFrame({'p': row['p'], 'split': row['split'], 'recall': RECALL_GRID,
                                                         'precision': c}) for row, c in results], ignore_index=True)
            done.sort_values(['p', 'split'], ignore_index=True).to_csv(PER_SPLIT, index=False)
            curves.sort_values(['p', 'split', 'recall'], ignore_index=True).to_csv(CURVES, index=False)
    return done.sort_values(['p', 'split'], ignore_index=True), curves


def summary(done, curves, q, lfc):
    """spot_split.summarise_de over each p_split's replicates, in spot_split's results table and PR-curve data. Each
    replicate's precision at RECALL_GRID stands in for its curve, which summarise_de's interpolation returns
    unchanged."""
    import kidney.spot_split as ss
    by_p = {}
    for p, d in done.groupby('p'):
        reps = []
        for r in d.itertuples():
            c = curves[np.isclose(curves['p'].astype(float), p) & (curves['split'] == r.split)].sort_values('recall')
            reps.append({'MAST': {k: getattr(r, k) for k in METRICS}
                         | {'precision_curve': c['precision'].to_numpy(), 'recall_curve': ss.RECALL_GRID}})
        by_p[p] = ss.summarise_de(reps)
    p_values = np.array(sorted(by_p))
    ss.save_results_de(p_values, by_p, RESULTS, q, lfc)
    ss.save_pr_curve_data(p_values, by_p, CURVE_JSON, q, lfc)


def combined(df, curve_data, mast_df, mast_curve_data):
    """spot_split's with-DEG results and PR curves with MAST's beside them, for spot_split_plots' draw functions."""
    curves = {k: v | mast_curve_data['curves'].get(k, {}) for k, v in curve_data['curves'].items()}
    return pd.concat([df, mast_df], ignore_index=True), dict(curve_data, curves=curves)


def plot(df, curve_data):
    from kidney import spot_split_plots as ssp
    from method_colors import method_draw_order
    plot_style.use()
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2))
    ssp.draw_tpr_fpr(axes, df, SPLIT)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    out = OUT / fig_name('split_degs_mast_tpr_fpr')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')
    p_values, curves = np.array(curve_data['p_values']), curve_data['curves']
    methods = method_draw_order(curves[next(iter(curves))])
    n_cols = min(ssp.GRID_COLS, len(p_values))
    n_rows = (len(p_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_style.figsize(n_cols, n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, p in zip(axes, p_values):
        ssp.draw_pr_curve(ax, curves, p, methods, SPLIT)
    ssp.label_grid_outer(list(axes[:len(p_values)]), n_cols)
    for ax in axes[len(p_values):]:
        ax.axis('off')
    plt.tight_layout()
    plot_style.legend_outside(fig)
    out = OUT / fig_name('split_degs_mast_pr_curves_grid')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    parser = argparse.ArgumentParser(description="MAST on the kidney's with-DEG spot splitting")
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSVs')
    args = parser.parse_args()
    df = pd.read_csv(require_input(DEGS, what='the with-DEG spot-splitting results',
                                   source='python -m kidney.spot_split (see README)'))
    curve_data = json.load(open(require_input(DEG_CURVES, what='their PR curves',
                                              source='python -m kidney.spot_split (see README)')))
    q, lfc = float(df['q'].iloc[0]), float(df['lfc'].iloc[0])
    if not args.plot_only:
        null = pd.read_csv(require_input(NULL_SPLITS, what='the no-DEG spot-splitting splits, whose seeds these share',
                                         source='python -m kidney.spot_split (see README)'))
        seeds = null[(null['method'] == 'LN_test') & (null['split'] < SPLITS)][['p', 'split', 'seed']]
        summary(*run(seeds, q, lfc), q, lfc)
    mast_df = pd.read_csv(require_input(RESULTS, what="MAST's with-DEG summary",
                                        source='this script without --plot-only'))
    mast_curves = json.load(open(require_input(CURVE_JSON, what="MAST's PR curves",
                                               source='this script without --plot-only')))
    plot(*combined(df, curve_data, mast_df, mast_curves))


if __name__ == '__main__':
    main()
