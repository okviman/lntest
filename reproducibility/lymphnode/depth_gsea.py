#!/usr/bin/env python3
"""Lymph node, the T-cell zone's reads thinned and the B-cell follicles whole: does each test's Hallmark enrichment of
the B-cell follicles against the T-cell zone keep agreeing with the pseudo-bulk reference's as the T-cell zone's variance
grows?

The conditions, cells, filters and tests of lymphnode/depth_imbalance.py, its replicates 0 to REPLICATES - 1 (the same
draws; the unthinned data once). Each test's score ranking of the B-cell follicles against the
T-cell zone (all tested genes; LN's score, as decision-ln-gsea-ranking recommends) and the reference's Wald statistic,
oriented B over T (and, alongside, its log2 fold change), go through clustering.concordance.prerank (MSigDB Hallmark
2020, 1,000 permutations, seed 42), in this process one ranking at a time. Readout per task and method: the Spearman
correlation between its NES and the reference's. Each task's depth ratio and scores must reproduce depth_imbalance.py's
results.csv.

    python -m lymphnode.depth_gsea --config lymphnode/config_50rep.yaml               # from reproducibility/, minutes
    python -m lymphnode.depth_gsea --config lymphnode/config_50rep.yaml --thin bins   # the spot version
    python -m lymphnode.depth_gsea --config lymphnode/config_50rep.yaml --plot-only
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'   # one thread per worker, as lymphnode/subsampling.py

import argparse
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plot_style
from clustering.concordance import prerank
from evaluation_utils import _get_rank_genes_df
from lymphnode import depth_imbalance as di
from lymphnode.subsampling import load_config, reference_markers
from paths import require_input, results_dir

REPLICATES = 10
OUT = results_dir(__file__) / 'depth_gsea'
OUTS = {'reads': OUT, 'bins': results_dir(__file__) / 'depth_gsea_bins'}
REFERENCE = {'stat': 'reference_stat', 'log2FoldChange': 'reference_lfc'}


def task(args):
    """One condition and replicate: depth_imbalance's scores, and each method's ranking of the B-cell follicles."""
    adata = di.tested(*args)
    ranks = {m: _get_rank_genes_df(adata, di.B_FOLLICLES, key)[['names', 'scores']].astype({'scores': float})
             for m, key in di.KEYS.items()}
    return di.score(adata, *args), ranks


def reference_sets(path):
    """The pseudo-bulk reference's enrichment, ranked by its Wald statistic and by its log2 fold change, B over T."""
    ref = pd.read_csv(path)
    sign = np.sign(ref.loc[ref['marker_of'] == di.B_FOLLICLES, 'log2FoldChange'].median())
    return pd.concat([prerank(pd.DataFrame({'names': ref['gene'].astype(str), 'value': sign * ref[col].astype(float)}))
                      .assign(thinned='reference', p=np.nan, replicate=0, ranking=name)
                      for col, name in REFERENCE.items()], ignore_index=True)


def run(config, n_workers, thin='reads'):
    """Every condition and replicate in a fixed order; the tests in fresh workers, the enrichments here in order (in
    parallel processes gseapy's results differed between runs)."""
    markers = reference_markers(config['reference'])
    filters = {k: config[k] for k in ('min_spots_per_cell', 'min_counts_after_subsampling', 'min_cells_per_gene')}
    adata_path = require_input(config['adata_path'], what='the Visium HD lymph node matrix',
                               source='bash lymphnode/reproduce_vishd_cluster1_cluster3.sh')
    sets = [reference_sets(config['reference'])]
    tasks = di.tasks(REPLICATES, thin)
    ctx = multiprocessing.get_context('spawn')
    rows = []
    # a fresh worker per task, as depth_imbalance.py
    with ctx.Pool(n_workers, initializer=di.init, maxtasksperchild=1,
                  initargs=(str(adata_path), config['cluster_column'], config['cell_id_column'], markers, filters,
                            thin)) as pool:
        for k, ((thinned, p, replicate), (scored, ranks)) in enumerate(zip(tasks, pool.imap(task, tasks, chunksize=1))):
            rows += scored
            for m, r in ranks.items():
                sets.append(prerank(r).assign(thinned=thinned or 'none', p=p, replicate=replicate, ranking=m))
            if (k + 1) % 10 == 0:
                print(f'  {k + 1}/{len(tasks)} tasks', flush=True)
    return pd.DataFrame(rows), pd.concat(sets, ignore_index=True)


def check(scored, results_path):
    """The tasks' depth ratios and scores against depth_imbalance.py's results.csv."""
    ref = pd.read_csv(require_input(results_path, what='the depth-imbalance results',
                                    source='python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml'))
    key = ['thinned', 'p', 'replicate', 'cluster', 'method']
    m = scored.merge(ref, on=key, suffixes=('', '_ref'))
    cols = ('calls', 'cells', 'genes', 'precision', 'recall', 'auprc', 'depth_ratio')
    bad = np.zeros(len(m), dtype=bool)
    for c in cols:
        bad |= ~np.isclose(m[c].astype(float), m[f'{c}_ref'].astype(float), equal_nan=True, rtol=0, atol=1e-12)
    if len(m) != len(scored) or bad.any():
        print(m.loc[bad, key + [x for c in cols for x in (c, f'{c}_ref')]].to_string(index=False))
        raise SystemExit('the tasks do not reproduce depth_imbalance results.csv; nothing written')


def agreement(scored, sets):
    """Per task and method, the Spearman correlation of its NES with the reference's, by each reference ranking."""
    ref = {name: sets[sets['ranking'] == name].set_index('set')['nes'] for name in REFERENCE.values()}
    depth = scored.drop_duplicates(['thinned', 'p', 'replicate']).set_index(['thinned', 'p', 'replicate'])['depth_ratio']
    rows = []
    for (thinned, p, replicate, m), d in sets[sets['thinned'] != 'reference'].groupby(
            ['thinned', 'p', 'replicate', 'ranking'], sort=False):
        nes = d.set_index('set')['nes']
        row = {'thinned': thinned, 'p': p, 'replicate': replicate, 'method': m,
               'depth_ratio': depth[thinned, p, replicate]}
        for name, r in ref.items():
            both = nes.index.intersection(r.index)
            row[name] = spearmanr(nes[both], r[both]).statistic
            row[f'{name}_sets'] = len(both)
        rows.append(row)
    a = pd.DataFrame(rows).sort_values(['thinned', 'p', 'replicate', 'method'], ignore_index=True)
    s = a.groupby(['thinned', 'p', 'method'])[['depth_ratio', *REFERENCE.values()]].mean().reset_index()
    return a, s


LABELS = {'reference_stat': 'NES agreement with\npseudo-bulk (Spearman)',
          'reference_lfc': 'NES agreement with pseudo-\nbulk fold change (Spearman)'}


def draw_agreement(ax, agreement, column='reference_stat', thin='reads', ylabel=None):
    """Each task's NES agreement with the reference as each method's mean and SD at each condition, by depth_imbalance's
    draw_errorbar."""
    di.draw_errorbar(ax, agreement, column, LABELS[column] if ylabel is None else ylabel, di.XLABEL[thin])
    ax.set_ylim(0, 1)


def plot(agreement, thin='reads'):
    plot_style.use()
    for column in REFERENCE.values():
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_agreement(ax, agreement, column, thin)
        di.save_with_key(fig, f"depth_gsea_{column.removeprefix('reference_')}", OUTS[thin])


def main():
    parser = argparse.ArgumentParser(description="Each test's Hallmark enrichment against the pseudo-bulk reference's, "
                                                 'one lymph node compartment thinned')
    parser.add_argument('--config', required=True, help='The subsampling config (lymphnode/config_50rep.yaml)')
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--thin', choices=list(OUTS), default='reads', help='Thin the T-cell zone\'s reads or its spots')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    out = OUTS[args.thin]
    out.mkdir(parents=True, exist_ok=True)
    csv = {k: out / f'{k}.csv' for k in ('sets', 'agreement', 'summary')}
    if args.plot_only:
        a = pd.read_csv(require_input(csv['agreement'], what='the depth-GSEA agreement',
                                      source='this script without --plot-only'))
    else:
        scored, sets = run(load_config(args.config), args.n_workers, args.thin)
        check(scored, di.OUTS[args.thin] / 'results.csv')
        a, summary = agreement(scored, sets)
        sets = sets[['thinned', 'p', 'replicate', 'ranking', 'set', 'nes', 'fdr']]
        for k, t in (('sets', sets), ('agreement', a), ('summary', summary)):
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
        print(summary.pivot_table(index=['thinned', 'p'], columns='method', values='reference_stat').round(3).to_string())
    plot(a, args.thin)


if __name__ == '__main__':
    main()
