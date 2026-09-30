"""The full-depth negative control: capsules split at the median library size, no downsampling.

Healthy capsules differ in depth but not in biology, so every DEG called between the deeper and the
shallower half is a false positive. The split is umi_null's at p_sample = 1: with all 161 capsules, and
again after removing the 20 and the 35 with the lowest total UMI, as umi_null's --n_cells_remove does.

    python -m kidney.full_depth      # from reproducibility/, a few seconds
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import statsmodels.stats.multitest as smm

import plot_style
from baselines import scanpy_sig_test
from lntest import get_LN_lfcs
from method_colors import method_color, method_label, method_order
from paths import data_dir, fig_name, require_input, results_dir

CORRECTIONS = {'BH': ('fdr_bh', 'benjamini-hochberg'), 'Bonferroni': ('bonferroni', 'bonferroni')}
SHOWN = 'Bonferroni'   # the correction the figures plot, as every experiment's
REMOVED = (0, 20, 35)


def load():
    h5ad = sc.read_h5ad(require_input(
        data_dir(__file__) / 'merged_blobs_in_cluster_5.h5ad', what='the per-capsule count matrix',
        source='python -m kidney.preprocess'))
    umis = h5ad.X.toarray() if sp.issparse(h5ad.X) else np.asarray(h5ad.X)
    return umis[:, umis.sum(0) > 0]


def split_by_depth(umis, n_removed):
    umis = umis[np.argsort(umis.sum(1))[n_removed:]]
    library_sizes = umis.sum(1)
    deep = library_sizes >= np.median(library_sizes)
    X, Y = umis[deep], umis[~deep]
    expressed = (X.sum(0) > 0) & (Y.sum(0) > 0)
    return X[:, expressed], Y[:, expressed]


def count_degs(X, Y, n_removed):
    rows = []
    _, p_ln = get_LN_lfcs(Y, X, test='t')
    for correction, (statsmodels_name, scanpy_name) in CORRECTIONS.items():
        adj = {'LN': smm.multipletests(p_ln, alpha=0.05, method=statsmodels_name)[1]}
        for method in ('t-test', 'Wilcoxon'):
            adj[method] = np.asarray(scanpy_sig_test(X, Y, method=method.lower(), corr_method=scanpy_name)[1])
        for method, a in adj.items():
            rows.append({'n_removed': n_removed, 'method': method, 'correction': correction,
                         'n_degs': int((a < 0.05).sum()),
                         'n_genes': X.shape[1], 'n_deep': X.shape[0], 'n_shallow': Y.shape[0]})
    return pd.DataFrame(rows)


def plot(df, out):
    d = df[(df['correction'] == SHOWN) & (df['n_removed'] == 0)].set_index('method')
    methods = method_order(d.index)
    fig, ax = plt.subplots(figsize=plot_style.figsize())
    ax.bar(range(len(methods)), d.loc[methods, 'n_degs'], width=0.6,
           color=[method_color(m) for m in methods], edgecolor='none')
    for k, m in enumerate(methods):
        ax.text(k, d.loc[m, 'n_degs'], f"{d.loc[m, 'n_degs']:,}", ha='center', va='bottom', fontsize=plot_style.SMALL)
    ax.set_xticks(range(len(methods)), [method_label(m).replace(' $t$', '\n$t$') for m in methods])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_ylabel('Number of DEGs')
    plot_style.boxplot_grid(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_by_removal(df, out):
    """SHOWN's counts for each capsule set, one group of method bars per number of capsules removed."""
    d = df[df['correction'] == SHOWN]
    methods = method_order(d['method'].unique())
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=plot_style.figsize(1.6))  # nine bars need more than one panel's width
    for k, m in enumerate(methods):
        dm = d[d['method'] == m].set_index('n_removed').loc[list(REMOVED)]
        x = np.arange(len(REMOVED)) + (k - (len(methods) - 1) / 2) * width
        ax.bar(x, dm['n_degs'], width=width * 0.9, color=method_color(m), edgecolor='none')
        for xi, n in zip(x, dm['n_degs']):
            ax.text(xi, n, f'{n:,}', ha='center', va='bottom', fontsize=plot_style.SMALL)
    kept = d.groupby('n_removed')[['n_deep', 'n_shallow']].first().sum(1)
    ax.set_xticks(range(len(REMOVED)), [f'{r}\n({kept[r]} kept)' for r in REMOVED])
    ax.set_xlabel('Smallest capsules removed')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_ylabel('Number of DEGs')
    plot_style.boxplot_grid(ax)
    fig.tight_layout()
    plot_style.legend_outside(fig, [Patch(facecolor=method_color(m)) for m in methods],
                              [method_label(m) for m in methods])
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_style.use()
    umis = load()
    df = pd.concat([count_degs(*split_by_depth(umis, r), r) for r in REMOVED], ignore_index=True)
    csv = results_dir(__file__) / 'full_depth_null.csv'
    df.to_csv(csv, index=False)
    print(df.to_string(index=False))
    for draw, stem in ((plot, 'full_depth_null'), (plot_by_removal, 'full_depth_null_by_removal')):
        out = results_dir(__file__) / fig_name(stem)
        draw(df, out)
        print(f'wrote {out}')
    print(f'wrote {csv}')
