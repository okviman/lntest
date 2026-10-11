#!/usr/bin/env python3
"""What does each method's ranked list look like as subclustering keeps breaking a cell type apart?

Each type clustering/celltype_labels.py includes with at least MIN_TYPE cells is subclustered recursively: a cluster of
at least MIN_PARENT cells is clustered within itself as clustering/de.py clusters, at the lowest of RESOLUTIONS that
splits it, and each child is tested against its siblings with the three tests as clustering/de.py runs them, over the
genes detected in at least MIN_DETECTED of the parent's cells; children of at least MIN_PARENT cells are split in turn,
to MAX_LEVEL. Per child and method: the mean statistic of the top TOP genes, in either direction and as markers,
against a standard normal's over as many genes; the overlap of the top TOP markers with LN's; what the top TOP are made
of; and the calls.

    python -m clustering.subcluster_recursive               # from reproducibility/, minutes
    python -m clustering.subcluster_recursive --plot-only   # redraw from the CSVs
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm

import plot_style
from clustering.replication import DATASETS, METHODS, load, run_tests, save
from clustering.subcluster_de import MIN_DETECTED, NEAR, TOP, counts, neighbours
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

MIN_TYPE = 400
MIN_PARENT = 200
MAX_LEVEL = 5
RESOLUTIONS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
MIN_SUMMARY = 100        # children summarised on their own from this many cells
OUT = results_dir(__file__) / 'subclustering' / 'recursive'


def split(parent):
    """The resolution and Leiden labels of the parent's first split along RESOLUTIONS, or None and None."""
    h = neighbours(parent)
    for r in RESOLUTIONS:
        sc.tl.leiden(h, resolution=r, key_added='sub')
        if h.obs['sub'].nunique() > 1:
            return r, h.obs['sub'].astype(str).to_numpy()
    return None, None


def noise(n):
    """The mean of the TOP largest absolute values, and of the TOP largest values, of n standard normal statistics
    (expected order statistics)."""
    i = np.arange(1, TOP + 1) - 0.5
    return float(norm.ppf(1 - i / (2 * n)).mean()), float(norm.ppf(1 - i / n).mean())


def readouts(res, ratio, absent):
    """One method's statistic for one child: the mean absolute statistic of its TOP strongest genes and the mean
    statistic of its TOP markers; the shares of the strongest that are absent on one side, have means within NEAR,
    or whose means differ against the statistic; the statistic's mean and SD; the calls; and the markers' names."""
    s = res['scores'].to_numpy(dtype=float)
    q = res['pvals_adj'].to_numpy(dtype=float)
    finite = np.isfinite(s)
    v = np.where(finite, s, 0.0)
    strongest = np.argsort(-np.abs(v), kind='stable')[:TOP]
    markers = np.argsort(-np.where(finite, s, -np.inf), kind='stable')[:TOP]
    with np.errstate(divide='ignore'):
        lr = np.log(ratio[strongest])
    return {'top_abs': np.abs(v[strongest]).mean(), 'top_up': s[markers].mean(),
            'absent': absent[strongest].mean(), 'near': (np.abs(lr) < np.log1p(NEAR)).mean(),
            'against': (np.sign(lr) * np.sign(v[strongest]) < 0).mean(),
            'mean_score': s[finite].mean(), 'sd_score': s[finite].std(),
            'up_calls': int(((q < 0.05) & (s > 0)).sum()), 'down_calls': int(((q < 0.05) & (s < 0)).sum())}, \
        set(res.index[markers])


def test_children(parent, labels):
    """Each child against its siblings: per child, its share of the parent and its readouts per method."""
    keep = np.asarray((parent.layers['counts'] > 0).sum(0)).ravel() >= MIN_DETECTED
    a = parent[:, keep].copy()
    a.obs['sub'] = pd.Categorical(labels)
    run_tests(a, 'sub')
    x, det = a.layers['norm_counts'].tocsr(), (a.layers['counts'] > 0).tocsr()
    noise_abs, noise_up = noise(a.n_vars)
    out = {}
    for c in a.obs['sub'].cat.categories:
        own = labels == c
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.asarray(x[own].mean(0)).ravel() / np.asarray(x[~own].mean(0)).ravel()
        absent = (np.asarray(det[own].sum(0)).ravel() == 0) | (np.asarray(det[~own].sum(0)).ravel() == 0)
        rows, tops = {}, {}
        for m in METHODS:
            res = _get_rank_genes_df(a, c, m).set_index('names').reindex(a.var_names)
            rows[m], tops[m] = readouts(res, ratio, absent)
        for m in METHODS:
            rows[m]['overlap_ln'] = len(tops[m] & tops['ln']) / TOP if m != 'ln' else np.nan
        out[c] = {'genes': a.n_vars, 'noise_abs': noise_abs, 'noise_up': noise_up, 'methods': rows}
    return out


def run():
    """Every dataset's recursive splits, breadth first; one row per child and method."""
    rows = []
    for name in DATASETS:
        adata = load(name)
        adata.layers['counts'] = counts(name, adata.obs_names)
        depth = np.asarray(adata.layers['counts'].sum(1)).ravel()
        sizes = adata.obs['type'].value_counts()
        queue = [(t, np.flatnonzero((adata.obs['type'] == t).to_numpy()), 1, '')
                 for t in sorted(sizes[sizes >= MIN_TYPE].index)]
        while queue:
            t, cells, level, path = queue.pop(0)
            if level > MAX_LEVEL or len(cells) < MIN_PARENT:
                continue
            parent = adata[cells].copy()
            r, labels = split(parent)
            if labels is None:
                print(f'  {name}, {t} {path or "root"}: {len(cells)} cells, no split')
                continue
            print(f'  {name}, {t} {path or "root"}: {len(cells)} cells, {len(np.unique(labels))} children at {r}')
            d = depth[cells]
            for c, res in test_children(parent, labels).items():
                own = labels == c
                key = {'dataset': name, 'type': t, 'level': level, 'path': path + c, 'resolution': r,
                       'cells': int(own.sum()), 'siblings': int((~own).sum()),
                       'depth_ratio': np.median(d[own]) / np.median(d[~own]), 'genes': res['genes'],
                       'noise_abs': res['noise_abs'], 'noise_up': res['noise_up']}
                rows += [{**key, 'method': m, **v} for m, v in res['methods'].items()]
                queue.append((t, cells[own], level + 1, path + c + '.'))
    return pd.DataFrame(rows)


def summarise(children):
    """Per dataset, method and level, over the children of at least MIN_SUMMARY cells and over all: their number, the
    median of each top TOP mean statistic over the noise level, the mean shares of the strongest TOP that are absent on
    one side or have nearly equal means (and against), the mean overlap with LN's markers, and the median calls."""
    c = children.assign(top_abs_noise=children['top_abs'] / children['noise_abs'],
                        top_up_noise=children['top_up'] / children['noise_up'],
                        technical=children['absent'] + children['near'])
    rows = []
    for subset, d in (('all', c), (f'{MIN_SUMMARY}+', c[c['cells'] >= MIN_SUMMARY])):
        g = d.groupby(['dataset', 'method', 'level'])
        s = pd.DataFrame({'children': g.size(), 'top_abs_noise': g['top_abs_noise'].median(),
                          'top_up_noise': g['top_up_noise'].median(), 'technical': g['technical'].mean(),
                          'absent': g['absent'].mean(), 'near': g['near'].mean(), 'against': g['against'].mean(),
                          'overlap_ln': g['overlap_ln'].mean(), 'up_calls': g['up_calls'].median(),
                          'down_calls': g['down_calls'].median()})
        rows.append(s.reset_index().assign(cells=subset))
    return pd.concat(rows, ignore_index=True)


def draw_levels(ax, summary, name, column, ylabel, methods=METHODS, noise_line=False):
    """One summary column against the recursion level for the children of at least MIN_SUMMARY cells, one line per
    method; the noise level dashed where the column is relative to it."""
    d = summary[(summary['dataset'] == name) & (summary['cells'] == f'{MIN_SUMMARY}+')]
    levels = sorted(d['level'].unique())
    for m in method_draw_order(methods):
        y = d[d['method'] == m].set_index('level').reindex(levels)[column]
        ax.plot(levels, y, marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                lw=plot_style.ERRORBAR['lw'], label=method_label(m), color=method_color(m))
    if noise_line:
        ax.axhline(1, color='0.4', lw=0.8, ls='--')
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1)
    n = d[d['method'] == METHODS[0]].set_index('level').reindex(levels)['children']
    ax.set_xticks(levels)
    ax.set_xticklabels([f'{lv}\n({k:.0f})' for lv, k in zip(levels, n)])
    ax.set_xlabel('Level (children)')
    ax.set_ylabel(ylabel)
    ax.set_title(name)


def depth_term(children, name):
    """A dataset's children of at least MIN_SUMMARY cells against siblings of at least as many: per method, each
    child's mean statistic over all genes and its depth term, log2 of its depth ratio times the square root of its
    effective size n1 n2 / (n1 + n2)."""
    d = children[(children['dataset'] == name) & (children['cells'] >= MIN_SUMMARY) & (children['siblings'] >= MIN_SUMMARY)]
    n_eff = d['cells'] * d['siblings'] / (d['cells'] + d['siblings'])
    return d.assign(depth_term=np.log2(d['depth_ratio']) * np.sqrt(n_eff))


def draw_depth_term(ax, children, name='PBMC10k', methods=METHODS):
    """Each child's mean statistic against its depth term, one colour per method, with its least-squares line and
    slope."""
    d = depth_term(children, name)
    lo, hi = d['depth_term'].min(), d['depth_term'].max()
    for m in method_draw_order(methods):
        e = d[d['method'] == m]
        slope, intercept = np.polyfit(e['depth_term'], e['mean_score'], 1)
        ax.scatter(e['depth_term'], e['mean_score'], s=6, lw=0, color=method_color(m), label=method_label(m))
        ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi], color=method_color(m), lw=0.8, ls='--')
    for k, m in enumerate(methods):
        e = d[d['method'] == m]
        ax.text(0.03, 0.97 - 0.09 * k, f'{method_label(m)}: slope {np.polyfit(e["depth_term"], e["mean_score"], 1)[0]:.3f}',
                transform=ax.transAxes, va='top', fontsize=plot_style.SMALL, color=method_color(m))
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xlabel('Depth term, log2(depth ratio) √n_eff')
    ax.set_ylabel('Mean statistic over genes')


def plot(summary, children):
    """The arm's figures: along the recursion, the top TOP statistics over the noise level, what the strongest are made
    of, and the baselines' overlap with LN's markers."""
    plot_style.use()
    panels = (('top_abs_noise', f'Top {TOP} |statistic|\nover noise', METHODS, True, 'recursive_top_statistic'),
              ('top_up_noise', f'Top {TOP} markers\' statistic\nover noise', METHODS, True, 'recursive_top_markers'),
              ('technical', f'Top {TOP}: absent on one side\nor means within {NEAR:.0%}', METHODS, False,
               'recursive_top_made_of'),
              ('overlap_ln', f'Top {TOP} markers\nshared with LN\'s', METHODS[1:], False, 'recursive_overlap'))
    for column, ylabel, methods, noise_line, stem in panels:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=plot_style.figsize(len(DATASETS)))
        for ax, name in zip(axes, DATASETS):
            draw_levels(ax, summary, name, column, ylabel, methods, noise_line)
        save(fig, OUT / fig_name(stem))
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_depth_term(ax, children)
    save(fig, OUT / fig_name('recursive_depth_term'))


def main():
    parser = argparse.ArgumentParser(description='Each method\'s ranked list along recursive subclustering of each type')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('children', 'summary')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the recursive subclustering {k}',
                                               source='this script without --plot-only')) for k, p in csv.items()}
    else:
        children = run()
        tables = {'children': children, 'summary': summarise(children)}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['summary'], tables['children'])


if __name__ == '__main__':
    main()
