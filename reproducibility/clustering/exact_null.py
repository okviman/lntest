#!/usr/bin/env python3
"""An exact null built from the arm's cells (PBMC3k, or PBMC10k with its config): every call is a false positive for a
test of equal means.

Each cluster of at least MIN_CELLS cells at RESOLUTION is split into random halves, and one half is binomially thinned
to each of FRACTIONS of its counts. Thinning keeps every cell's expected composition, so every gene's expected CP10K is
equal in the two halves; only depth, and with it sparsity and CP10K variance, differs. The CSV counts, per test, each
method's genes with uncorrected p below each of ALPHAS, with no p, and called under Bonferroni.

    python -m clustering.exact_null --config clustering/config.yaml --n-workers 4                  # from reproducibility/
    python -m clustering.exact_null --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k --n-workers 4
    python -m clustering.exact_null --config clustering/config.yaml --plot-only                     # redraw from the CSV
"""

import argparse
import os
import resource
import sys
from multiprocessing import Pool

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

import plot_style
from clustering.heldout_fp import ALPHAS, de_output, uncorrected
from evaluation_utils import _get_rank_genes_df
from lntest import rank_genes_groups_ln
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

METHODS = {'ln': None, 't_test': {'method': 't-test'}, 'wilcoxon': {'method': 'wilcoxon', 'tie_correct': True}}
RESOLUTION = 'leiden_mid'
MIN_CELLS = 200
SEEDS = (0, 1, 2)
FRACTIONS = (0.25, 0.5, 0.75)
MIN_DETECTED = 3         # a gene is tested if detected in this many of the cluster's cells
CSV = 'exact_null_thinned.csv'
SAMPLE = r'$p_\mathrm{sample}$'   # the thinned half's share of its counts, as the kidney's UMI downsampling names it
RSS_LIMIT_GB = 2.2

_DATA = None


def guard():
    """Peak memory of this process in GB; raises past RSS_LIMIT_GB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024) / 1e9
    if rss > RSS_LIMIT_GB:
        raise RuntimeError(f'peak RSS {rss:.2f} GB, over {RSS_LIMIT_GB} GB')
    return rss


def load_labels(adata_path):
    """The RESOLUTION cluster labels, without the rest of the h5ad."""
    with h5py.File(adata_path, 'r') as f:
        return ad.io.read_elem(f['obs'])[RESOLUTION].astype(str).to_numpy()


def load(adata_path):
    """The raw counts (CSR, as stored) and the RESOLUTION labels."""
    with h5py.File(adata_path, 'r') as f:
        counts = sp.csr_matrix(ad.io.read_elem(f['layers/counts']))
    return counts, load_labels(adata_path)


def detected(counts):
    """The genes detected in at least MIN_DETECTED of the cells."""
    return np.asarray((counts > 0).sum(0)).ravel() >= MIN_DETECTED


def run_tests(counts, grp, methods=METHODS):
    """Each method's test of group A against B on raw counts (cells x genes): LN on CP10K, scanpy's tests on log1p
    CP10K, Bonferroni; CP10K over the given genes. Returns the CP10K matrix and, per method, pvals, pvals_adj and
    scores in the genes' order."""
    sub = ad.AnnData(counts.astype(np.float32))
    sub.var_names = np.arange(counts.shape[1]).astype(str)
    sub.obs['grp'] = pd.Categorical(grp, categories=['A', 'B'])
    norm = sub.copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    # copy: csr_matrix() shares data, and log1p below works in place
    sub.layers['norm_counts'] = sp.csr_matrix(norm.X, copy=True)
    sc.pp.log1p(norm)
    sub.layers['log1p_norm'] = sp.csr_matrix(norm.X, copy=True)
    assert np.allclose(np.asarray(sub.layers['norm_counts'].sum(1)).ravel(), 1e4, rtol=1e-4)
    out = {}
    for m, kw in methods.items():
        if kw is None:
            rank_genes_groups_ln(sub, 'grp', groups=['A'], reference='B', layer='norm_counts', sparse=True,
                                 key_added=m, corr_method='bonferroni')
        else:
            sc.tl.rank_genes_groups(sub, 'grp', groups=['A'], reference='B', layer='log1p_norm', use_raw=False,
                                    corr_method='bonferroni', key_added=m, **kw)
        out[m] = _get_rank_genes_df(sub, 'A', m).set_index('names').reindex(sub.var_names)
    return sub.layers['norm_counts'], out


def thin(counts_deep, depth_target, rng):
    """Binomially thin each row of counts_deep to depth_target (same row order)."""
    c = counts_deep.tocsr().astype(np.int64)
    lib = np.asarray(c.sum(1)).ravel()
    p = np.minimum(1.0, depth_target / lib)
    rows = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    data = rng.binomial(c.data, p[rows])
    return sp.csr_matrix((data, c.indices, c.indptr), shape=c.shape)


def gap(counts, norm, grp):
    """How the halves differ: the median CV^2 ratio A/B over genes expressed in both, the difference in zero fractions,
    and the median depth ratio A/B."""
    a, b = norm[grp == 'A'], norm[grp == 'B']
    ma, mb = np.asarray(a.mean(0)).ravel(), np.asarray(b.mean(0)).ravel()
    va = np.asarray(a.multiply(a).mean(0)).ravel() - ma ** 2
    vb = np.asarray(b.multiply(b).mean(0)).ravel() - mb ** 2
    ok = (ma > 0) & (mb > 0)
    return {'median_cv2_ratio': float(np.median((va[ok] / ma[ok] ** 2) / (vb[ok] / mb[ok] ** 2))),
            'zero_frac_diff': float((1 - (a > 0).mean()) - (1 - (b > 0).mean())),
            'depth_ratio': float(np.median(np.asarray(counts[grp == 'A'].sum(1)))
                                 / np.median(np.asarray(counts[grp == 'B'].sum(1))))}


def tally(res, common):
    """One test's row: common, then per method the genes with p below each alpha, with no p, called under Bonferroni,
    and called with A higher."""
    row = dict(common)
    for m, r in res.items():
        p = r['pvals'].to_numpy(dtype=float)
        called = np.nan_to_num(r['pvals_adj'].to_numpy(dtype=float), nan=1.0) < 0.05
        row.update({uncorrected(m, a): int((p < a).sum()) for a in ALPHAS})
        row.update({f'{m}_nan': int(np.isnan(p).sum()), f'{m}_bonf': int(called.sum()),
                    f'{m}_bonf_a_higher': int((called & (r['scores'].to_numpy(dtype=float) > 0)).sum())})
    return row


def thinned(counts, labels, cl, seed, methods=METHODS):
    """Every fraction for cluster cl: random halves, half A thinned and tested against half B."""
    idx = np.flatnonzero(labels == cl)
    cc = counts[idx]
    cc = cc[:, detected(cc)]
    g = np.array(['A', 'B'])[np.random.default_rng(1000 + seed).permutation(len(idx)) % 2]
    lib = np.asarray(cc.sum(1)).ravel()
    a_rows = np.flatnonzero(g == 'A')
    grp = np.r_[np.repeat('A', len(a_rows)), np.repeat('B', (g == 'B').sum())]
    rows = []
    for f in FRACTIONS:
        both = sp.vstack([thin(cc[a_rows], f * lib[a_rows], np.random.default_rng(2000 + seed)), cc[g == 'B']]).tocsr()
        norm, res = run_tests(both, grp, methods)
        rows.append(tally(res, {'fraction': f, 'resolution': RESOLUTION, 'cluster': cl, 'n_cells': len(idx), 'seed': seed,
                                'n_a': len(a_rows), 'n_b': int((g == 'B').sum()), 'genes': cc.shape[1],
                                **gap(both, norm, grp)}))
    return rows


def _init(adata_path):
    global _DATA
    _DATA = load(adata_path)


def _thinned(args):
    rows = thinned(*_DATA, *args)
    guard()
    return rows


def run(adata_path, n_workers):
    """Every test's row, each cluster and seed in its own task, in a fixed order."""
    sizes = pd.Series(load_labels(adata_path)).value_counts()
    work = [(cl, seed) for cl in sorted(sizes.index[sizes >= MIN_CELLS], key=int) for seed in SEEDS]
    rows = []
    with Pool(n_workers, initializer=_init, initargs=(adata_path,)) as pool:
        for k, r in enumerate(pool.imap(_thinned, work)):
            rows += r
            print(f'  {k + 1}/{len(work)} clusters and seeds', flush=True)
    return pd.DataFrame(rows)


def shares(df, by, alpha):
    """Each method's share of genes with uncorrected p below alpha, total below over total genes tested, per group."""
    columns = [uncorrected(m, alpha) for m in METHODS]
    totals = df.groupby(by)[['genes', *columns]].sum()
    return totals[columns].div(totals['genes'], axis=0).set_axis(list(METHODS), axis=1)


def draw_shares(ax, s, alpha, xlabel, zero=None, top=None):
    """shares()'s rows against their index, one line per method, on a log axis with a zero row; alpha dashed."""
    values = s.to_numpy().ravel()
    if zero is None:
        zero = plot_style.zero_row(values, np.zeros_like(values))
    if top is None:
        top = float(np.nanmax(values))
    for m in method_draw_order(s.columns):
        y = s[m].to_numpy()
        ax.plot(s.index, np.where(y == 0, zero, y), marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                lw=plot_style.ERRORBAR['lw'], label=method_label(m), color=method_color(m))
    ax.axhline(alpha, color='0.4', lw=0.8, ls='--', zorder=1)
    plot_style.log_axis_with_zero(ax, zero, max(top, alpha))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Fraction with p < {alpha:g}')


def draw_thinned(ax, df, alpha=0.05, zero=None, top=None):
    """The pooled share against the thinned half's share of its counts."""
    s = shares(df, 'fraction', alpha)
    draw_shares(ax, s, alpha, SAMPLE, zero, top)
    ax.set_xticks(s.index)


def plot_alphas(draw, path):
    """draw(ax, alpha) for each of ALPHAS, side by side, with one key."""
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=plot_style.figsize(len(ALPHAS)))
    for ax, alpha in zip(axes, ALPHAS):
        draw(ax, alpha)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser(description="An exact null built from the arm's cells: thinned random halves")
    parser.add_argument('--config', type=str, required=True, help='Path to the clustering YAML config')
    parser.add_argument('--output-dir', type=str, default=str(results_dir(__file__)),
                        help='Where clustering/de.py wrote its h5ad; the CSV goes to its metrics/, the figure to figures/')
    parser.add_argument('--n-workers', type=int, default=4, help='Processes, one cluster and seed at a time each')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figure from the saved CSV')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    metrics_dir, figures_dir = (os.path.join(args.output_dir, d) for d in ('metrics', 'figures'))
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    csv = os.path.join(metrics_dir, CSV)
    if args.plot_only:
        df = pd.read_csv(require_input(csv, what='the thinned null', source='this script without --plot-only'))
    else:
        adata_path = require_input(de_output(config['adata_path'], args.output_dir), what='the DE-annotated AnnData',
                                   source='run `python -m clustering.de` first')
        df = run(adata_path, args.n_workers)
        df.to_csv(csv, index=False)
        print(f'  wrote {csv}')
    plot_style.use()
    plot_alphas(lambda ax, alpha: draw_thinned(ax, df, alpha), os.path.join(figures_dir, fig_name('exact_null_thinned')))


if __name__ == '__main__':
    main()
