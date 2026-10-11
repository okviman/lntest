#!/usr/bin/env python3
"""Each test's calibration from uncorrected p-values, on the kidney's two null experiments.

Spot splitting (kidney/spot_split.py without DEGs): each capsule's spots split at random into two semi-capsules,
probability p for the first; downsampling (kidney/umi_null.py): every capsule's counts thinned binomially by p, then
split by median depth. Every gene is null in both. Per replicate, the share of tested genes whose uncorrected p is
below each of ALPHAS; a calibrated test gives alpha. The groups are built as the two arms build them, with their
settings (35 capsules removed, their p grids, seed 0), and the tests run as their fpr_test_single runs them (genes
with no counts in either group dropped; LN on the counts, scanpy's t-test and Wilcoxon on CP10K log1p), only the
p-values are read before correction. Writes its own CSV, so Fig 1's results are untouched. The figures draw each
experiment's share against p, the mean and SD over replicates, one panel per alpha, alpha dashed.

    python -m kidney.null_calibration              # from reproducibility/, tens of minutes on all cores
    python -m kidney.null_calibration --plot-only  # redraw from the CSV
"""

import argparse
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import plot_style
from baselines import scanpy_sig_test
from kidney.fpr_plots import SAMPLING, SPLIT
from kidney.spot_split import create_groups_from_shape_ids
from lntest import get_LN_lfcs
from method_colors import method_color, method_draw_order, method_label
from paths import data_dir, fig_name, require_input, results_dir

ALPHAS = (0.05, 0.01, 0.001)
REMOVE = 35          # capsules removed, as both arms are run (README)
N_REPS = 100
SPLIT_P = np.linspace(0.1, 0.5, 10)       # spot_split's defaults
DOWNSAMPLE_P = np.linspace(0.1, 1.0, 10)  # umi_null's defaults
METHODS = {'LN_test': None, 'Scanpy t-test': 't-test', 'Scanpy wilcoxon': 'wilcoxon'}
CSV = 'null_calibration.csv'
FIGURES = {'spot splitting': ('split_null_calibration', SPLIT), 'downsampling': ('downsample_null_calibration', SAMPLING)}

_DATA = None


def _init(data):
    global _DATA
    _DATA = data


def shares(X, Y):
    """Per method, the share of genes with uncorrected p below each alpha, genes absent from either group dropped."""
    keep = (X.sum(0) > 0) & (Y.sum(0) > 0)
    X, Y = X[:, keep], Y[:, keep]
    out = {}
    for name, method in METHODS.items():
        if method is None:
            _, p = get_LN_lfcs(Y, X, test='t')
        else:
            _, _, p = scanpy_sig_test(X, Y, method=method, return_raw=True)
        p = np.asarray(p, dtype=float)
        out[name] = {a: float(np.mean(p < a)) for a in ALPHAS} | {'n_genes': int(keep.sum())}
    return out


def split_replicate(args):
    p, seed = args
    umis, index, ids = _DATA
    X, Y = create_groups_from_shape_ids(umis, index, ids, p, seed=seed)
    return shares(X, Y)


def downsample_replicate(args):
    p, seed = args
    (umis,) = _DATA
    np.random.seed(seed)
    thinned = np.random.binomial(umis.astype(int), p).astype(float) if p < 1.0 else umis.copy()
    depth = thinned.sum(1)
    median = np.median(depth)
    return shares(thinned[depth >= median], thinned[depth < median])


def spots():
    """podocytes_2um.h5ad as spot_split.py loads it, without its REMOVE lowest-count capsules."""
    h5ad = sc.read_h5ad(require_input(data_dir(__file__) / 'podocytes_2um.h5ad', what='the per-2um-spot matrix',
                                      source='kidney/preprocess.ipynb'))
    umis, ids = sp.csr_matrix(h5ad.X), h5ad.obs['shape_id'].values
    totals = {s: umis[ids == s].sum() for s in np.unique(ids)}
    removed = {s for s, _ in sorted(totals.items(), key=lambda kv: kv[1])[:REMOVE]}
    keep = np.array([s not in removed for s in ids])
    umis, ids = umis[keep].copy(), ids[keep]
    unique = np.unique(ids)
    return umis, {s: np.where(ids == s)[0] for s in unique}, unique


def capsules():
    """merged_blobs_in_cluster_5.h5ad as umi_null.py loads it, without its REMOVE lowest-count capsules."""
    h5ad = sc.read_h5ad(require_input(data_dir(__file__) / 'merged_blobs_in_cluster_5.h5ad',
                                      what='the merged glomerular-capsule matrix', source='kidney/preprocess.ipynb'))
    umis = h5ad.X.toarray() if sp.issparse(h5ad.X) else np.asarray(h5ad.X)
    umis = umis[:, umis.sum(0) > 0]
    return (umis[np.argsort(umis.sum(1))[REMOVE:]].copy(),)


def run(experiment, data, replicate, p_values, n_jobs):
    """Every replicate at every p, seeded as the arms seed theirs."""
    np.random.seed(0)
    rows = []
    with Pool(n_jobs, initializer=_init, initargs=(data,)) as pool:
        for p in p_values:
            seeds = np.random.randint(0, 2**31, size=N_REPS)
            for rep, res in enumerate(pool.map(replicate, [(p, s) for s in seeds])):
                for method, r in res.items():
                    rows += [{'experiment': experiment, 'p': round(float(p), 4), 'replicate': rep, 'method': method,
                              'n_genes': r['n_genes'], 'alpha': a, 'share': r[a]} for a in ALPHAS]
            print(f'  {experiment} p={p:.3f} done', flush=True)
    return rows


def draw_calibration(ax, df, experiment, alpha=0.05, zero=None, top=None):
    """One experiment's share of genes with uncorrected p below alpha against p, the mean and SD over replicates, per
    method, on the kidney's log axis with a zero row; alpha dashed."""
    d = df[(df['experiment'] == experiment) & np.isclose(df['alpha'], alpha)]
    s = d.groupby(['method', 'p'])['share'].agg(['mean', 'std']).reset_index()
    if zero is None:
        zero = plot_style.zero_row(s['mean'], s['std'])
    if top is None:
        top = float((s['mean'] + s['std']).max())
    for m in method_draw_order(s['method'].unique()):
        r = s[s['method'] == m].sort_values('p')
        plot_style.errorbar_log(ax, r['p'], r['mean'], r['std'], zero, label=method_label(m), color=method_color(m))
    ax.axhline(alpha, color='0.4', lw=0.8, ls='--', zorder=1)
    plot_style.log_axis_with_zero(ax, zero, max(top, alpha))
    ax.set_xlabel(FIGURES[experiment][1])
    ax.set_ylabel(f'Fraction with p < {alpha:g}')


def plot(df, experiment, path):
    """draw_calibration for each of ALPHAS, side by side, with one key."""
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=plot_style.figsize(len(ALPHAS)))
    for ax, alpha in zip(axes, ALPHAS):
        draw_calibration(ax, df, experiment, alpha)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser(description="Calibration from uncorrected p-values on the kidney's null experiments")
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSV')
    args = parser.parse_args()
    out = results_dir(__file__) / CSV
    if args.plot_only:
        df = pd.read_csv(require_input(out, what='the kidney null calibration', source='this script without --plot-only'))
    else:
        rows = run('spot splitting', spots(), split_replicate, SPLIT_P, args.n_jobs)
        rows += run('downsampling', capsules(), downsample_replicate, DOWNSAMPLE_P, args.n_jobs)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f'wrote {out}')
    plot_style.use()
    for experiment, (stem, _) in FIGURES.items():
        plot(df, experiment, results_dir(__file__) / fig_name(stem))


if __name__ == '__main__':
    main()
