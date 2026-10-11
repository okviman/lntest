"""Each test's effect sizes in PBMC10k's and PBMC3k's multi-resolution clustering, as each package reports them.

Every Leiden cluster against the rest at five resolutions: the mean |LFC| over the test's significant genes (Bonferroni
0.05, clustering.de). LN's t-test's LFC estimate, the log1p t-test's LFC as Scanpy reports it (logfoldchanges, the
log2 ratio of expm1 of the mean log1p CP10K) and the difference the log1p t-test tests (the log2 difference of the mean
log1p CP10K, the arm's "actual LFC") come from clustering.metrics' avg_lfc_by_resolution.csv; the Wilcoxon's as
Seurat 5's FindMarkers reports it (avg_log2FC: log2 of the summed CP10K plus 1, over the cells, cluster against rest) is
computed here from the DE output's norm_counts layer, for the Wilcoxon's significant genes. With each test's number of
significant genes per cluster, the genes those means are over (clustering.metrics' n_sig_genes_by_resolution.csv), drawn
by draw_degs in the same slots as draw_effect_sizes, so the two line up.

    python -m clustering.effect_sizes              # from reproducibility/, seconds; both datasets
    python -m clustering.effect_sizes --plot-only  # redraw from the CSVs
"""

import argparse
import textwrap

import anndata as ad
import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

import plot_style
from clustering.celltype_labels import dataset_paths
from clustering.heldout_fp import de_output
from method_colors import MEDIANPROPS, method_color, method_label
from paths import fig_name, require_input, results_dir

NAMES = ('PBMC10k', 'PBMC3k')
OUT = results_dir(__file__) / 'effect_sizes'
CSVS = {name: OUT / f'{name.lower()}.csv' for name in NAMES}
ARM = {'LN test': 'ln', 'Scanpy t-test (log1p)': 't_test', 'Scanpy t-test (actual LFC)': 't_test_log1p',
       'Scanpy Wilcoxon (log1p)': 'wilcoxon'}   # the arm's labels
SOURCE = {'ln': "LN's estimate", 't_test': "Scanpy's LFC", 't_test_log1p': 'difference of log1p means',
          'wilcoxon': "Seurat's LFC"}
ORDER = ('ln', 't_test', 't_test_log1p', 'wilcoxon')
TEST = {'t_test_log1p': 't_test'}   # whose colour and name a series takes
ALPHA = {'t_test_log1p': 0.45}       # lighter, as Fig 2f's LN score rank
YLABEL = 'Mean |LFC| of\nsignificant genes'


def seurat_lfc(x, labels):
    """Seurat 5's avg_log2FC of every cluster against the rest, clusters by genes: log2 of the summed CP10K plus 1, over
    the cells (FoldChange.Assay's log1pdata.mean.fxn, pseudocount.use = 1)."""
    clusters = np.asarray(sorted(pd.unique(labels), key=lambda c: int(c) if str(c).isdigit() else str(c)))
    row = {c: i for i, c in enumerate(clusters)}
    member = sp.csr_matrix((np.ones(len(labels)), ([row[c] for c in labels], np.arange(len(labels)))),
                           shape=(len(clusters), len(labels)))
    sums = np.asarray((member @ x).todense())
    n = np.asarray(member.sum(axis=1)).ravel()[:, None]
    rest = sums.sum(axis=0) - sums
    return clusters, np.log2((sums + 1) / n) - np.log2((rest + 1) / (len(labels) - n))


def run(name):
    config_path, out = dataset_paths()[name]
    config = yaml.safe_load(open(config_path))
    keys = [k for _, k in config['leiden_resolutions']]
    path = require_input(de_output(config['adata_path'], str(out)), what=f'the {name} DE output',
                         source='python -m clustering.de')
    with h5py.File(path, 'r') as f:
        obs = ad.io.read_elem(f['obs'])
        genes = np.asarray(ad.io.read_elem(f['var']).index)
        x = sp.csr_matrix(ad.io.read_elem(f['layers/norm_counts']))
        wilcoxon = {k: ad.io.read_elem(f[f'uns/wilcoxon_{k}']) for k in keys}
    index = pd.Index(genes)
    rows = []
    for k in keys:
        clusters, lfc = seurat_lfc(x, obs[k].astype(str).to_numpy())
        for i, c in enumerate(clusters):
            names, padj = wilcoxon[k]['names'][c], wilcoxon[k]['pvals_adj'][c]
            sig = index.get_indexer(names[padj < 0.05])
            rows.append({'resolution': k, 'clusters': len(clusters), 'cluster': c, 'method': 'wilcoxon',
                         'mean_abs_lfc': np.abs(lfc[i, sig]).mean() if len(sig) else np.nan})
    arm = pd.read_csv(require_input(out / 'metrics' / 'avg_lfc_by_resolution.csv', what=f"{name}'s reported LFCs",
                                    source='python -m clustering.metrics (see README)'), dtype={'cluster': str})
    arm = arm[arm['method'].isin(list(ARM)) & (arm['method'] != 'Scanpy Wilcoxon (log1p)')]   # the Wilcoxon's is Seurat's
    n = obs[keys].nunique()
    arm = pd.DataFrame({'resolution': arm['resolution'], 'clusters': arm['resolution'].map(n), 'cluster': arm['cluster'],
                        'method': arm['method'].map(ARM), 'mean_abs_lfc': arm['avg_abs_lfc_sig']})
    df = pd.concat([arm, pd.DataFrame(rows)], ignore_index=True)
    degs = pd.read_csv(require_input(out / 'metrics' / 'n_sig_genes_by_resolution.csv', what=f"{name}'s DEG counts",
                                     source='python -m clustering.metrics (see README)'), dtype={'cluster': str})
    degs = degs.assign(method=degs['method'].map(ARM)).rename(columns={'n_sig_genes': 'n_degs'})
    df = df.merge(degs[['resolution', 'cluster', 'method', 'n_degs']], on=['resolution', 'cluster', 'method'], how='left')
    df['source'] = df['method'].map(SOURCE)
    df['order'] = df['cluster'].astype(int)
    return df.sort_values(['clusters', 'method', 'order'], ignore_index=True).drop(columns='order')


def series(methods):
    """The series present, in the key's order."""
    return [m for m in ORDER if m in set(methods)]


def colour(m):
    return method_color(TEST.get(m, m)), ALPHA.get(m, 1.0)


def key(methods, wrap=False):
    """The key's handles and labels, each test with whose LFC it shows."""
    source = (lambda m: textwrap.fill(SOURCE[m], 14)) if wrap else SOURCE.get
    return ([plt.Rectangle((0, 0), 1, 1, facecolor=colour(m)[0], edgecolor='none', alpha=colour(m)[1]) for m in series(methods)],
            [f"{method_label(TEST.get(m, m))}{chr(10) if wrap else ', '}{source(m)}" for m in series(methods)])


def value_range(tables, column='mean_abs_lfc'):
    """A log y range holding every table's values, for plots that share one."""
    v = pd.concat([t[column] for t in tables]).dropna()
    return v.min() / 1.4, v.max() * 1.4


def draw_boxes(ax, df, column, ylabel, ylim=None):
    """A column's per-cluster values as a box at each resolution, every cluster a point over its box, every series in its
    own slot whether or not it has values here, so plots of different columns line up; on a log axis, on ylim if given."""
    n = df.groupby('clusters')['clusters'].first().sort_values()
    methods = series(df['method'].unique())
    width = 0.84 / len(methods)
    rng = np.random.default_rng(0)
    for j, m in enumerate(methods):
        for i, c in enumerate(n):
            v = df.loc[(df['clusters'] == c) & (df['method'] == m), column].dropna().to_numpy()
            if not len(v):
                continue
            x = i + (j - (len(methods) - 1) / 2) * width
            bp = ax.boxplot([v], positions=[x], widths=width * 0.85, patch_artist=True, showfliers=False,
                            medianprops=MEDIANPROPS, manage_ticks=False)
            for box in bp['boxes']:
                box.set_facecolor(colour(m)[0])
                box.set_alpha(colour(m)[1])
                box.set_edgecolor('none')
            c, a = colour(m)
            plot_style.strip(ax, x, v, mcolors.to_rgb(c) if a == 1 else tuple(a * np.array(mcolors.to_rgb(c)) + 1 - a),
                             width, rng)   # a lighter series' points as light as its box
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.to_numpy())
    ax.set_xlim(-0.5, len(n) - 0.5)
    ax.set_yscale('log')   # the values span orders of magnitude
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(ylabel)
    plot_style.boxplot_grid(ax)


def draw_effect_sizes(ax, df, ylim=None):
    """Each series' per-cluster mean |LFC| over the test's significant genes."""
    draw_boxes(ax, df, 'mean_abs_lfc', YLABEL, ylim)


def draw_degs(ax, df, ylim=None):
    """Each test's significant genes per cluster, in the slots draw_effect_sizes uses."""
    draw_boxes(ax, df, 'n_degs', 'DEGs per cluster', ylim)


def count_key(methods):
    """The tests' key for the DEG counts."""
    tests = [m for m in series(methods) if m not in TEST]
    return ([plt.Rectangle((0, 0), 1, 1, facecolor=colour(m)[0], edgecolor='none') for m in tests],
            [method_label(m) for m in tests])


def plot(df, name):
    plot_style.use()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=plot_style.figsize(1, 2), sharex=True)
    draw_degs(top, df)
    draw_effect_sizes(bottom, df)
    top.set_xlabel('')
    top.set_title(name)
    plot_style.legend_outside(fig, *key(df['method'].unique()))
    out = OUT / fig_name(f'{name.lower()}_effect_sizes')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    parser = argparse.ArgumentParser(description="Each test's effect sizes in PBMC10k's and PBMC3k's clustering")
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSVs')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for name, csv in CSVS.items():
        if args.plot_only:
            df = pd.read_csv(require_input(csv, what=f'the {name} effect sizes', source='this script without --plot-only'),
                             dtype={'cluster': str})
        else:
            df = run(name)
            df.to_csv(csv, index=False)
            print(f'  wrote {csv}')
        plot(df, name)


if __name__ == '__main__':
    main()
