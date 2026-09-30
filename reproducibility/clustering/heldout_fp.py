#!/usr/bin/env python3
"""False positives in the arm's clusters, on negatives chosen from held-out cells (PBMC3k, or PBMC10k with its config).

Each split divides the cells at random into a selection half and a test half, within each cluster at the finest
resolution, so every cluster at every resolution is in both. On the selection half, a cluster's negatives are the genes
expressed in both it and the rest whose mean CP10K expression differs by less than a tolerance, binned by how much
their variances differ. On the test half every test runs as clustering/de.py runs it, and a negative it calls
significant (either direction) is a false positive. The CSV counts calls three ways: Bonferroni-adjusted p < 0.05, the
figures' default, since its cutoff is the same for every test; BH-adjusted p < 0.05, whose cutoff loosens with the
number of genes a test calls, so that its rate on negatives grows with the test's power; and uncorrected p below each
of ALPHAS. The negatives come from cells no test sees, so they do not depend on any test's statistic. Genes within the
tolerance can still differ slightly, and a test with the power to see it is counted as wrong, so the rates are upper
bounds for every method.

    python -m clustering.heldout_fp --config clustering/config.yaml               # from reproducibility/, about a minute
    python -m clustering.heldout_fp --config clustering/config.yaml --plot-only   # redraw from the CSV
"""

import argparse
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

import plot_style
from evaluation_utils import _get_rank_genes_df
from lntest import rank_genes_groups_ln
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

SPLITS = 5
TOLERANCES = (0.05, 0.1, 0.2)   # |log2 fold change| of the means; 0.1 is within 7%
TOLERANCE = 0.1
EDGES = (0, 0.5, 1, 1.5, 2, 3, np.inf)   # |log2 variance ratio|
BINS = ('1–1.4', '1.4–2', '2–2.8', '2.8–4', '4–8', '>8')
SIZES = {'<100 cells': (0, 100), '100–400 cells': (100, 401), '>400 cells': (401, np.inf)}
METHODS = ('ln', 't_test', 'wilcoxon', 'wilcoxon_untied')
MAIN = ('ln', 't_test', 'wilcoxon')      # Wilcoxon without tie correction, scanpy's current default, is an ED check
XLABEL = 'Variance ratio, cluster vs rest'
CSV = 'heldout_fp.csv'
ALPHAS = (0.05, 0.01, 0.001)             # uncorrected p-value cutoffs; a calibrated test calls these shares
CORRECTION = 'bonferroni'                # the figures' default: the same cutoff for every test
CORRECTIONS = ('bonferroni', 'bh', 0.05)  # 'bonferroni', 'bh', or an uncorrected alpha


def tolerance_title(tolerance):
    """A tolerance as the largest difference in means it allows."""
    return f'Means within {2 ** tolerance - 1:.1%}'


def de_output(adata_path, output_dir):
    """The h5ad clustering/de.py writes for this input: its name, less the extension, with _DE.h5ad."""
    name = os.path.basename(adata_path.rstrip('/'))
    for ext in ('.tar.gz', '.tgz', '.h5ad'):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    else:
        name = os.path.splitext(name)[0]
    return os.path.join(output_dir, f'{name}_DE.h5ad')


def selection_half(labels, seed):
    """Half of each cluster's cells, at random."""
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(labels), bool)
    for idx in pd.Series(np.arange(len(labels))).groupby(labels).groups.values():
        sel[rng.permutation(np.asarray(idx))[: len(idx) // 2]] = True
    return sel


def run_tests(adata, key):
    """Every test on adata's clusters under key, as clustering/de.py runs them, plus Wilcoxon without tie correction.
    BH-adjusted, so that one run counts BH calls; Bonferroni and uncorrected calls come from the raw p-values."""
    rank_genes_groups_ln(adata, key, sparse=True, key_added=f'ln_{key}', layer='norm_counts',
                         corr_method='benjamini-hochberg')
    for method, name, tie in (('t-test', 't_test', False), ('wilcoxon', 'wilcoxon', True),
                              ('wilcoxon', 'wilcoxon_untied', False)):
        sc.tl.rank_genes_groups(adata, groupby=key, method=method, corr_method='benjamini-hochberg',
                                key_added=f'{name}_{key}', layer='log1p_norm', use_raw=False, tie_correct=tie)


def compute(adata, keys):
    """Negatives and false positives per split, resolution, cluster, tolerance and variance-ratio bin."""
    base = ad.AnnData(X=adata.layers['counts'], obs=adata.obs[keys].astype(str).astype('category'),
                      var=pd.DataFrame(index=adata.var_names))
    for layer in ('norm_counts', 'log1p_norm'):
        base.layers[layer] = adata.layers[layer]
    norm = adata.layers['norm_counts']
    norm = (norm.toarray() if sp.issparse(norm) else np.asarray(norm)).astype(np.float64)
    genes = adata.var_names
    rows = []
    for split in range(SPLITS):
        sel = selection_half(base.obs[keys[-1]].to_numpy(), split)
        test = base[~sel].copy()
        for key in keys:
            test.obs[key] = test.obs[key].cat.remove_unused_categories()
            run_tests(test, key)
            labels = base.obs[key].to_numpy()
            for cl in test.obs[key].cat.categories:
                c, r = sel & (labels == cl), sel & (labels != cl)
                mc, mr = norm[c].mean(0), norm[r].mean(0)
                vc, vr = norm[c].var(0, ddof=1), norm[r].var(0, ddof=1)
                expressed = (mc > 0) & (mr > 0) & (vc > 0) & (vr > 0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    lfc, ratio = np.abs(np.log2(mc / mr)), np.abs(np.log2(vr / vc))
                vbin = np.digitize(ratio, EDGES[1:-1])
                res = {m: _get_rank_genes_df(test, cl, f'{m}_{key}').set_index('names').reindex(genes) for m in METHODS}
                called = {m: res[m]['pvals_adj'].fillna(1).to_numpy() < 0.05 for m in METHODS}
                raw = {m: res[m]['pvals'].fillna(1).to_numpy() for m in METHODS}
                bonferroni = {m: raw[m] < 0.05 / len(genes) for m in METHODS}
                for tol in TOLERANCES:
                    neg = expressed & (lfc < tol)
                    for b in range(len(BINS)):
                        in_bin = neg & (vbin == b)
                        rows.append({'split': split, 'resolution': key, 'cluster': cl, 'n_cells': int((labels == cl).sum()),
                                     'tolerance': tol, 'bin': BINS[b], 'negatives': int(in_bin.sum()),
                                     **{m: int(called[m][in_bin].sum()) for m in METHODS},
                                     **{column(m, 'bonferroni'): int(bonferroni[m][in_bin].sum()) for m in METHODS},
                                     **{uncorrected(m, a): int((raw[m][in_bin] < a).sum()) for m in METHODS for a in ALPHAS}})
        print(f'  split {split + 1}/{SPLITS} done')
    return pd.DataFrame(rows)


def uncorrected(method, alpha):
    """The CSV column counting a method's negatives with uncorrected p below alpha."""
    return f'{method}_p{alpha:g}'


def column(method, correction):
    """The CSV column counting a method's calls on negatives: under 'bonferroni' or 'bh', or at an uncorrected alpha."""
    if correction == 'bonferroni':
        return f'{method}_bonf'
    if correction == 'bh':
        return method
    return uncorrected(method, correction)


def correction_title(correction):
    """How a correction is named on the figures."""
    return {'bonferroni': 'Bonferroni', 'bh': 'BH'}.get(correction) or f'Uncorrected, p < {correction:g}'


def summary(df, tolerance=TOLERANCE, size=None, correction=CORRECTION):
    """Each method's false-positive rate per bin, pooled over clusters: mean and SD over the splits."""
    d = df[df['tolerance'] == tolerance]
    if size is not None:
        lo, hi = SIZES[size]
        d = d[(d['n_cells'] >= lo) & (d['n_cells'] < hi)]
    columns = {m: column(m, correction) for m in METHODS}
    per_split = d.groupby(['split', 'bin'])[['negatives', *columns.values()]].sum()
    rates = per_split[list(columns.values())].div(per_split['negatives'], axis=0).rename(columns={v: k for k, v in columns.items()})
    return rates.groupby('bin').agg(['mean', 'std']).reindex(list(BINS))


def draw_fpr(ax, df, methods=MAIN, tolerance=TOLERANCE, size=None, zero=None, top=None, correction=CORRECTION):
    """False-positive rate against the variance ratio, one line per method, on the log axis Fig 1 uses."""
    s = summary(df, tolerance, size, correction)
    if zero is None:
        zero = plot_style.zero_row(s.xs('mean', axis=1, level=1)[list(methods)].values.ravel(),
                                   s.xs('std', axis=1, level=1)[list(methods)].values.ravel())
    if top is None:
        top = float((s.xs('mean', axis=1, level=1) + s.xs('std', axis=1, level=1))[list(methods)].max().max())
    x = np.arange(len(BINS))
    for m in method_draw_order(methods):
        plot_style.errorbar_log(ax, x, s[(m, 'mean')], s[(m, 'std')], zero, label=method_label(m),
                                color=method_color(m))
    plot_style.log_axis_with_zero(ax, zero, top)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel('FPR' if correction == CORRECTION else f'FPR ({correction_title(correction)})')


def shared_range(df, panels, methods):
    """One zero row and top for several panels, each given as summary()'s keyword arguments."""
    means, sds = [], []
    for kw in panels:
        s = summary(df, **kw)
        means.append(s.xs('mean', axis=1, level=1)[list(methods)].values.ravel())
        sds.append(s.xs('std', axis=1, level=1)[list(methods)].values.ravel())
    means, sds = np.concatenate(means), np.concatenate(sds)
    return plot_style.zero_row(means, sds), float(np.nanmax(means + sds))


def plot_panels(df, panels, titles, methods, path, ylabel=None):
    """A row of draw_fpr panels on one y range, with one key; ylabel in place of the first panel's own."""
    zero, top = shared_range(df, panels, methods)
    fig, axes = plt.subplots(1, len(panels), figsize=plot_style.figsize(len(panels)), sharey=True, squeeze=False)
    for ax, kw, title in zip(axes[0], panels, titles):
        draw_fpr(ax, df, methods, zero=zero, top=top, **kw)
        if title:
            ax.set_title(title)
    for ax in axes[0][1:]:
        ax.set_ylabel('')
    if ylabel:
        axes[0][0].set_ylabel(ylabel)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser(description='False positives on held-out negatives in the clusters')
    parser.add_argument('--config', type=str, required=True, help='Path to the clustering YAML config')
    parser.add_argument('--output-dir', type=str, default=str(results_dir(__file__)),
                        help='Where clustering/de.py wrote its h5ad; the CSV goes to its metrics/, the figures to figures/')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSV')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    keys = [r[1] for r in config['leiden_resolutions']]
    metrics_dir, figures_dir = (os.path.join(args.output_dir, d) for d in ('metrics', 'figures'))
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    csv = os.path.join(metrics_dir, CSV)
    if args.plot_only:
        df = pd.read_csv(require_input(csv, what='the held-out false positives', source='this script without --plot-only'))
    else:
        adata = ad.read_h5ad(require_input(de_output(config['adata_path'], args.output_dir),
                                           what='the DE-annotated AnnData', source='run `python -m clustering.de` first'))
        df = compute(adata, keys)
        df.to_csv(csv, index=False)
        print(f'  wrote {csv}')
    plot_style.use()
    plot_panels(df, [{}], [None], MAIN, os.path.join(figures_dir, fig_name('heldout_fpr')))
    plot_panels(df, [{'size': s} for s in SIZES], list(SIZES), METHODS,
                os.path.join(figures_dir, fig_name('heldout_fpr_by_size')))
    plot_panels(df, [{'tolerance': t} for t in TOLERANCES], [tolerance_title(t) for t in TOLERANCES],
                METHODS, os.path.join(figures_dir, fig_name('heldout_fpr_tolerance')))
    plot_panels(df, [{'correction': c} for c in CORRECTIONS], [correction_title(c) for c in CORRECTIONS],
                METHODS, os.path.join(figures_dir, fig_name('heldout_fpr_correction')), ylabel='FPR')


if __name__ == '__main__':
    main()
