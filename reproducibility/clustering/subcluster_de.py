#!/usr/bin/env python3
"""How many markers does each method find between subclusters of one annotated cell type?

Each type clustering/celltype_labels.py includes with at least MIN_CELLS cells in a dataset is subclustered as
clustering/de.py clusters (seurat_v3 HVGs on counts; CP10K, log1p and scaling on them; PCA; neighbours; Leiden at each
of RESOLUTIONS), and each subcluster is tested against the type's other cells with the three tests as clustering/de.py
runs them, over the genes detected in at least MIN_DETECTED of the type's cells. The types are taken as homogeneous,
so every marker (Bonferroni-significant, positive score) counts as a false positive. Each test's statistic is also
summarised: how often it reaches p < 0.05 on the genes whose CP10K means differ by less than NEAR, and the mean score of
its markers and of its strongest genes, each baseline's compared with LN's in the same subcluster.

    python -m clustering.subcluster_de               # from reproducibility/, minutes
    python -m clustering.subcluster_de --plot-only   # redraw from the CSVs
"""

import argparse

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

import plot_style
from clustering.celltype_labels import dataset_paths
from clustering.heldout_fp import de_output
from clustering.replication import DATASETS, METHODS, load, run_tests, save
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

MIN_CELLS = 200
MIN_DETECTED = 3
RESOLUTIONS = (0.25, 0.5, 1.0)
DEPTH_BINS = (0.67, 1.5)
TOP = 20
NEAR = 0.1               # means within this fraction of each other count as nearly equal
SUMMARIES = ['near_p05', 'near_bonf', 'marker_score', 'top_marker_score', 'sig_abs_score', 'top_abs_score']
SCORES = SUMMARIES[2:]
OUT = results_dir(__file__) / 'subclustering'


def counts(name, cells):
    """The raw counts of cells, over the arm's genes."""
    config_path, out = dataset_paths()[name]
    with h5py.File(de_output(yaml.safe_load(open(config_path))['adata_path'], str(out)), 'r') as f:
        index = ad.io.read_elem(f['obs']).index
        x = sp.csr_matrix(ad.io.read_elem(f['layers/counts']))
    return x[index.get_indexer(cells)]


def neighbours(parent):
    """The parent's highly variable genes with their neighbour graph, built as clustering/de.py builds it."""
    a = ad.AnnData(X=parent.layers['counts'].astype(np.float32), var=pd.DataFrame(index=parent.var_names))
    sc.pp.filter_genes(a, min_cells=MIN_DETECTED)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor='seurat_v3')
    h = a[:, a.var['highly_variable']].copy()
    sc.pp.normalize_total(h, target_sum=1e4)
    sc.pp.log1p(h)
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, svd_solver='arpack')
    sc.pp.neighbors(h, n_neighbors=10, n_pcs=40)
    return h


def subcluster(parent):
    """The parent's Leiden subclusters at each resolution, clustered as clustering/de.py clusters."""
    h = neighbours(parent)
    labels = {}
    for r in RESOLUTIONS:
        sc.tl.leiden(h, resolution=r, key_added='sub')
        labels[r] = h.obs['sub'].astype(str).to_numpy()
    return labels


def summaries(res, near):
    """One method's statistic in one subcluster's test: the genes significant with a negative score; the genes whose
    means are nearly equal, the share of them below p 0.05 and those significant; the mean score of the markers and of
    the top TOP; and the mean absolute score of the significant genes and of the TOP with the largest."""
    p, q, s = (res[k].to_numpy(dtype=float) for k in ('pvals', 'pvals_adj', 'scores'))
    sig = q < 0.05
    up = np.sort(s[sig & (s > 0)])[::-1]
    strongest = np.sort(np.abs(s[np.isfinite(s)]))[::-1][:TOP]
    return {'down': int((sig & (s < 0)).sum()), 'near_genes': int(near.sum()),
            'near_p05': float((p[near] < 0.05).mean()) if near.any() else np.nan, 'near_bonf': int((sig & near).sum()),
            'marker_score': up.mean() if len(up) else np.nan, 'top_marker_score': up[:TOP].mean() if len(up) else np.nan,
            'sig_abs_score': np.abs(s[sig]).mean() if sig.any() else np.nan, 'top_abs_score': strongest.mean()}


def test(parent, labels):
    """Each subcluster against the parent's other cells, over the genes detected in at least MIN_DETECTED of the
    parent's cells: the number of genes tested; per method and subcluster, the markers (Bonferroni-significant with a
    positive score), best first, and the summaries of its statistic."""
    genes = np.asarray((parent.layers['counts'] > 0).sum(0)).ravel() >= MIN_DETECTED
    a = parent[:, genes].copy()
    a.obs['sub'] = pd.Categorical(labels)
    run_tests(a, 'sub')
    x = a.layers['norm_counts'].tocsr()
    near = {}
    for c in a.obs['sub'].cat.categories:
        own = labels == c
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.asarray(x[own].mean(0)).ravel() / np.asarray(x[~own].mean(0)).ravel()
            near[c] = np.abs(np.log(ratio)) < np.log1p(NEAR)
    markers, stats = [], {}
    for m in METHODS:
        for c in a.obs['sub'].cat.categories:
            res = _get_rank_genes_df(a, c, m)
            sig = res[res['pvals_adj'] < 0.05]
            up = sig[sig['scores'] > 0].sort_values(['scores', 'names'], ascending=[False, True])
            markers.append(up.assign(method=m, subcluster=c, rank=np.arange(1, len(up) + 1)))
            stats[m, c] = summaries(res.set_index('names').reindex(a.var_names), near[c])
    return int(genes.sum()), pd.concat(markers, ignore_index=True), stats


def run():
    """Every dataset's subclusters, their calls and markers, and the summary."""
    subs, calls, marks = [], [], []
    for name in DATASETS:
        adata = load(name)
        adata.layers['counts'] = counts(name, adata.obs_names)
        depth = np.asarray(adata.layers['counts'].sum(1)).ravel()
        sizes = adata.obs['type'].value_counts()
        for t in sorted(sizes[sizes >= MIN_CELLS].index):
            inside = (adata.obs['type'] == t).to_numpy()
            parent, d = adata[inside].copy(), depth[inside]
            for r, labels in subcluster(parent).items():
                groups = np.unique(labels)
                if len(groups) < 2:
                    print(f'  {name}, {t}, resolution {r}: one subcluster, skipped')
                    continue
                genes, mk, stats = test(parent, labels)
                print(f'  {name}, {t}, resolution {r}: {len(groups)} subclusters, {genes} genes tested')
                key = {'dataset': name, 'type': t, 'resolution': r}
                for c in groups:
                    own = labels == c
                    subs.append({**key, 'subcluster': c, 'n_cells': int(own.sum()), 'n_siblings': int((~own).sum()),
                                 'depth_ratio': np.median(d[own]) / np.median(d[~own]), 'genes_tested': genes})
                    for m in METHODS:
                        calls.append({**key, 'subcluster': c, 'method': m,
                                      'markers': int(((mk['method'] == m) & (mk['subcluster'] == c)).sum()),
                                      **stats[m, c]})
                marks.append(mk.rename(columns={'names': 'gene'}).assign(**key))
    subs = pd.DataFrame(subs)
    calls = pd.DataFrame(calls).merge(subs[['dataset', 'type', 'resolution', 'subcluster', 'n_cells', 'depth_ratio']])
    columns = ['dataset', 'type', 'resolution', 'subcluster', 'method', 'rank', 'gene', 'scores', 'logfoldchanges',
               'pvals', 'pvals_adj']
    return {'subclusters': subs, 'calls': calls, 'markers': pd.concat(marks, ignore_index=True)[columns],
            'summary': summarise(calls), 'relative': relative(calls)}


def depth_bin(ratio):
    """The depth-ratio bin of each ratio: below, within or above DEPTH_BINS."""
    lo, hi = DEPTH_BINS
    return np.where(ratio < lo, f'below {lo}', np.where(ratio > hi, f'above {hi}', f'{lo}-{hi}'))


def summarise(calls):
    """Per dataset, method and resolution, and per dataset, method and depth-ratio bin: the subclusters, their markers
    in total and per subcluster, the shares of subclusters with at least one marker and with a full top TOP, and the
    means over subclusters of the statistic's summaries."""
    c = calls.assign(depth_bin=depth_bin(calls['depth_ratio']), any=calls['markers'] > 0,
                     full=calls['markers'] >= TOP)
    rows = []
    for by in ('resolution', 'depth_bin'):
        g = c.groupby(['dataset', 'method', by])
        s = pd.DataFrame({'subclusters': g.size(), 'markers': g['markers'].sum(), 'per_subcluster': g['markers'].mean(),
                          'with_any': g['any'].mean(), 'with_full_top': g['full'].mean(), 'down': g['down'].mean(),
                          **{k: g[k].mean() for k in SUMMARIES}})
        rows.append(s.reset_index().rename(columns={by: 'level'}).assign(by=by))
    return pd.concat(rows, ignore_index=True)[['dataset', 'method', 'by', 'level', 'subclusters', 'markers',
                                               'per_subcluster', 'with_any', 'with_full_top', 'down', *SUMMARIES]]


def ratios(calls):
    """Per subcluster and baseline, its score summaries over LN's, with the subcluster's depth ratio."""
    c = calls.set_index(['dataset', 'type', 'resolution', 'subcluster'])
    ln = c[c['method'] == 'ln'][SCORES]
    parts = []
    for m in METHODS[1:]:
        b = c[c['method'] == m]
        parts.append((b[SCORES] / ln).assign(method=m, depth_ratio=b['depth_ratio']))
    return pd.concat(parts).reset_index()


def relative(calls):
    """Each baseline's score summaries over LN's, per subcluster; their median per dataset, method and depth-ratio
    bin."""
    r = ratios(calls)
    g = r.assign(depth_bin=depth_bin(r['depth_ratio'].to_numpy())).groupby(['dataset', 'method', 'depth_bin'])
    out = g[SCORES].median().assign(subclusters=g.size()).reset_index()
    return out[['dataset', 'method', 'depth_bin', 'subclusters', *SCORES]]


def draw_by_resolution(ax, summary, name):
    """Markers per subcluster against the subclustering resolution, one line per method."""
    d = summary[(summary['by'] == 'resolution') & (summary['dataset'] == name)]
    d = d.assign(level=d['level'].astype(float))
    x = np.arange(len(RESOLUTIONS))
    for m in method_draw_order(METHODS):
        y = d[d['method'] == m].set_index('level').reindex(list(RESOLUTIONS))['per_subcluster']
        ax.plot(x, y, marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'],
                label=method_label(m), color=method_color(m))
    n = d[d['method'] == METHODS[0]].set_index('level').reindex(list(RESOLUTIONS))['subclusters']
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:g}\n({k:.0f})' for r, k in zip(RESOLUTIONS, n.fillna(0))])
    ax.set_xlabel('Resolution (subclusters)')
    ax.set_ylabel('Markers per subcluster')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_title(name)


def depth_points(ax, d, methods, column):
    """One point per subcluster, its column against its depth ratio, one colour per method; the edges of the
    depth-ratio bins dashed and the depth ratio on a log axis."""
    for m in method_draw_order(methods):
        e = d[d['method'] == m]
        ax.scatter(e['depth_ratio'], e[column], s=8, lw=0, alpha=0.7, color=method_color(m), label=method_label(m))
    for edge in DEPTH_BINS:
        ax.axvline(edge, color='0.4', lw=0.8, ls='--')
    ax.set_xscale('log')
    ticks = [0.25, 0.5, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.minorticks_off()


def draw_by_depth(ax, calls, name, column='markers', ylabel='Markers per subcluster'):
    """Each subcluster's count (or share, for a column of shares) against its depth ratio, every resolution
    together."""
    depth_points(ax, calls[calls['dataset'] == name], METHODS, column)
    if column in ('markers', 'down', 'near_bonf'):
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel('Depth, subcluster / siblings')
    ax.set_ylabel(ylabel)
    ax.set_title(name)


def draw_ratio(ax, calls, name, column, ylabel):
    """Each subcluster's baseline score summary over LN's against its depth ratio; equal statistics dashed."""
    r = ratios(calls)
    depth_points(ax, r[r['dataset'] == name], METHODS[1:], column)
    ax.axhline(1, color='0.4', lw=0.8, ls='--')
    ax.set_yscale('log')
    ticks = [0.5, 1, 2, 4]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel('Depth, subcluster / siblings')
    ax.set_ylabel(ylabel)
    ax.set_title(name)


def plot(summary, calls):
    """The arm's figures: markers per subcluster by resolution and against the depth ratio; against the depth ratio,
    the share of nearly-equal-mean genes below p 0.05 and each baseline's statistic over LN's."""
    plot_style.use()
    for draw, data, stem in ((draw_by_resolution, summary, 'subcluster_markers'),
                             (draw_by_depth, calls, 'subcluster_markers_by_depth')):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=plot_style.figsize(len(DATASETS)))
        for ax, name in zip(axes, DATASETS):
            draw(ax, data, name)
        save(fig, OUT / fig_name(stem))
    panels = ((draw_by_depth, 'near_p05', f'Fraction below p 0.05,\nmeans within {NEAR:.0%}', 'subcluster_near_equal'),
              (draw_ratio, 'top_abs_score', f'|Statistic| over LN\'s,\nstrongest {TOP}', 'subcluster_top_score_ratio'),
              (draw_ratio, 'top_marker_score', f'Statistic over LN\'s,\ntop {TOP} markers', 'subcluster_marker_score_ratio'))
    for draw, column, ylabel, stem in panels:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=plot_style.figsize(len(DATASETS)))
        for ax, name in zip(axes, DATASETS):
            draw(ax, calls, name, column, ylabel)
        save(fig, OUT / fig_name(stem))


def main():
    parser = argparse.ArgumentParser(description='Markers between subclusters of each annotated type, PBMC3k and PBMC10k')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('subclusters', 'calls', 'markers', 'summary', 'relative')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the subclustering {k}', source='this script without --plot-only'))
                  for k, p in csv.items()}
    else:
        tables = run()
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['summary'], tables['calls'])


if __name__ == '__main__':
    main()
