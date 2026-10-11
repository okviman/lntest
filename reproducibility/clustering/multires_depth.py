#!/usr/bin/env python3
"""Do each method's scores follow a cluster's sequencing depth, across Leiden resolutions?

Reads clustering/de.py's outputs (each cluster against all other cells, five resolutions, both datasets) and reruns no
test. Per comparison and method, the median score over the cluster's expressed genes (detected in at least EXPRESSED of
the cluster's or the rest's cells, as published/abdelfattah2022.py), against the cluster's log2 depth ratio, its median
counts per cell over the rest's (as multires_gaps.py, whose comparisons.csv it must reproduce); per dataset and method,
the Spearman and Pearson correlations and the slope over the comparisons.

    python -m clustering.multires_depth              # from reproducibility/; about a minute
    python -m clustering.multires_depth --plot-only  # redraw from the CSVs
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

import plot_style
from clustering import multires as mr
from clustering.celltype_labels import dataset_paths
from clustering.replication import save
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label, method_order
from paths import fig_name, require_input, results_dir

OUT = results_dir(__file__) / 'multires_depth'
GAPS = results_dir(__file__) / 'multires_gaps' / 'comparisons.csv'
EXPRESSED = 0.05
METHODS = mr.METHODS


def run():
    """Per comparison, its log2 depth ratio and each method's median score over its expressed genes."""
    rows = []
    for name in dataset_paths():
        adata, cp, _, keys, _ = mr.load(name)
        depth = adata.obs['n_counts'].to_numpy(dtype=float)
        detected = (cp > 0).tocsr()
        for key in keys:
            lab = adata.obs[key].astype(str).to_numpy()
            ids = sorted(np.unique(lab), key=lambda v: int(v) if v.isdigit() else v)
            for c in ids:
                own = lab == c
                expressed = ((np.asarray(detected[own].mean(0)).ravel() >= EXPRESSED)
                             | (np.asarray(detected[~own].mean(0)).ravel() >= EXPRESSED))
                genes = adata.var_names[expressed]
                row = {'dataset': name, 'resolution': key, 'clusters': len(ids), 'cluster': c, 'cells': int(own.sum()),
                       'depth_ratio': np.median(depth[own]) / np.median(depth[~own]),
                       'expressed_genes': int(expressed.sum())}
                for m in METHODS:
                    s = _get_rank_genes_df(adata, c, f'{m}_{key}').set_index('names')['scores'].astype(float)
                    row[m] = float(np.nanmedian(s.reindex(genes).to_numpy()))
                rows.append(row)
        print(f'  {name}: {sum(r["dataset"] == name for r in rows)} comparisons', flush=True)
    comps = pd.DataFrame(rows)
    comps['log2_depth_ratio'] = np.log2(comps['depth_ratio'])
    check = pd.read_csv(require_input(GAPS, what='the multires_gaps comparisons', source='python -m clustering.multires_gaps'),
                        dtype={'cluster': str})
    m = comps.merge(check, on=['dataset', 'resolution', 'cluster'], suffixes=('', '_gaps'))
    if len(m) != len(comps) or not np.allclose(m['depth_ratio'], m['depth_ratio_gaps']):
        raise SystemExit('the depth ratios do not reproduce multires_gaps/comparisons.csv; nothing written')
    return comps


def correlations(comps):
    rows = []
    for name, d in comps.groupby('dataset', sort=False):
        for m in METHODS:
            fit = linregress(d['log2_depth_ratio'], d[m])
            rows.append({'dataset': name, 'method': m, 'comparisons': len(d),
                         'spearman': spearmanr(d['log2_depth_ratio'], d[m]).statistic,
                         'pearson': pearsonr(d['log2_depth_ratio'], d[m]).statistic, 'slope': fit.slope})
    return pd.DataFrame(rows)


def draw_depth(ax, comps, name):
    """Each comparison's median score over its expressed genes against its log2 depth ratio, one colour per method, with
    its least-squares line and its Pearson r in the lower right."""
    d = comps[comps['dataset'] == name]
    x = d['log2_depth_ratio']
    span = np.array([x.min(), x.max()])
    for m in method_draw_order(METHODS):
        ax.scatter(x, d[m], s=6, lw=0, alpha=0.8, color=method_color(m), label=method_label(m))
        fit = linregress(x, d[m])
        ax.plot(span, fit.intercept + fit.slope * span, color=method_color(m), lw=1, ls='--')
    for k, m in enumerate(method_order(METHODS)):
        ax.text(0.97, 0.04 + 0.08 * k, f'r = {pearsonr(x, d[m]).statistic:.2f}', color=method_color(m),
                fontsize=plot_style.SMALL, ha='right', va='bottom', transform=ax.transAxes)
    ax.axhline(0, color='0.6', lw=0.6)
    ax.axvline(0, color='0.6', lw=0.6)
    ax.set_xlabel('log2 depth ratio')
    ax.set_ylabel('Median score')


def plot(comps):
    plot_style.use()
    for name in dataset_paths():
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_depth(ax, comps, name)
        ax.set_title(name)
        save(fig, OUT / fig_name(f'{name.lower()}_score_vs_depth'))


def main():
    parser = argparse.ArgumentParser(description="Each method's scores against the cluster's depth, across resolutions")
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('comparisons', 'correlations')}
    if args.plot_only:
        comps = pd.read_csv(require_input(csv['comparisons'], what='the multires_depth comparisons',
                                          source='this script without --plot-only'), dtype={'cluster': str})
    else:
        comps = run()
        tables = {'comparisons': comps, 'correlations': correlations(comps)}
        for k, v in tables.items():
            v.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
        print(tables['correlations'].round(3).to_string(index=False))
    plot(comps)


if __name__ == '__main__':
    main()
