#!/usr/bin/env python3
"""How well does each method's view of a cell type carry over from PBMC3k to PBMC10k: its top markers, its full score
ranking, its scores, and the GSEA enrichments of that ranking?

The cell types of clustering/replication.py are each tested against the rest of the included cells in both datasets,
the tests run as that script runs them, and each method's genes are ranked by its score. Per method and type: 1, the
replication arm's top-N concordance; 2, the Spearman correlation of the scores between the datasets over the genes
tested in both, and over the union of the two datasets' top TOP; 3, the Pearson correlation of the scores, and the
total-least-squares slope of PBMC10k's on PBMC3k's after dividing each comparison's by the square root of its effective
cell number; 4, gseapy.prerank on each dataset's full score ranking with the MSigDB Hallmark 2020 sets, and between the
datasets the Spearman correlation of the normalised enrichment scores and the agreement of the sets significant at FDR q
below FDR (significant in both with the same sign, over significant in either). A diagnostic repeats 2 and 4 with genes
ranked by LN's LFC and by scanpy's logfoldchanges, over the genes detected in both the type and the rest; a second
repeats 3 over the genes significant for every method in both datasets, over the genes well detected in both, and
over each method's own genes significant in both, all of them and its top EQUAL_K by their weaker score; the top
CEILING_K's scores in both datasets are kept for the ceiling plot.

    python -m clustering.concordance               # from reproducibility/, minutes
    python -m clustering.concordance --plot-only   # redraw from the CSVs
"""

import argparse
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import plot_style
from clustering import replication as rep
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

GMT = Path.home() / '.cache' / 'gseapy' / 'Enrichr.MSigDB_Hallmark_2020.gmt'
TOP = 100
N_MAIN = 10
PERMUTATIONS, SEED, MIN_SIZE, MAX_SIZE = 1000, 42, 15, 500      # as lymphnode/gsea_utils.py calls prerank
FDR = 0.25
OUT = results_dir(__file__) / 'concordance'


def tests():
    """Both datasets tested, with the replication arm's types, universes and markers; and each comparison's cells."""
    data = {name: rep.load(name) for name in rep.DATASETS}
    for adata in data.values():
        rep.run_tests(adata)
    types = sorted(data['PBMC3k'].obs['type'].cat.categories)
    sizes = {name: {t: int((a.obs['type'] == t).sum()) for t in types} for name, a in data.items()}
    return data, types, sizes


def markers(data, types):
    """1: the replication arm's pooled top-N concordance under the score rule, which must reproduce its CSV."""
    universe = {'PBMC3k': rep.annotation('PBMC10k'), 'PBMC10k': rep.annotation('PBMC3k')}
    table, _ = rep.marker_table(data, universe, types)
    pool = rep.pooled(rep.concordance(table))
    ref = pd.read_csv(require_input(rep.OUT / 'concordance_pooled.csv', what='the replication concordance',
                                    source='python -m clustering.replication'))
    keys = ['ranking', 'method', 'n']
    got, ref = (t.sort_values(keys).reset_index(drop=True) for t in (pool, ref))
    if not (got[keys].astype(str).equals(ref[keys].astype(str)) and np.allclose(got['concordance'], ref['concordance'])):
        raise SystemExit('the top-N concordance does not reproduce replication/concordance_pooled.csv; nothing written')
    return pool[pool['ranking'] == 'score'].drop(columns='ranking')


def scores(data, types, sizes):
    """2 and 3, per method and type, over the genes tested in both datasets with a finite score in both."""
    rows = []
    for t in types:
        n_eff = {name: 1 / (1 / sizes[name][t] + 1 / (data[name].n_obs - sizes[name][t])) for name in rep.DATASETS}
        for m in rep.METHODS:
            s = {name: _get_rank_genes_df(data[name], t, m).set_index('names')['scores'].astype(float)
                 for name in rep.DATASETS}
            both = pd.concat(s, axis=1, join='inner').replace([np.inf, -np.inf], np.nan).dropna()
            a, b = both['PBMC3k'], both['PBMC10k']
            top = set(a.nlargest(TOP).index) | set(b.nlargest(TOP).index)
            rows.append({'type': t, 'method': m, 'genes': len(both), 'ranking_spearman': spearmanr(a, b).statistic,
                         'top_union': len(top), 'top_spearman': spearmanr(a[list(top)], b[list(top)]).statistic,
                         'score_pearson': pearsonr(a, b).statistic,
                         'score_slope': rep.tls_slope(a / np.sqrt(n_eff['PBMC3k']), b / np.sqrt(n_eff['PBMC10k']))})
    return pd.DataFrame(rows)


def prerank(r):
    """gseapy.prerank on a ranking (names, value), best first; each Hallmark set's NES and FDR q."""
    r = r[np.isfinite(r.iloc[:, 1])].sort_values([r.columns[1], 'names'], ascending=[False, True])
    res = gp.prerank(rnk=r.reset_index(drop=True), gene_sets=str(GMT), permutation_num=PERMUTATIONS, seed=SEED,
                     min_size=MIN_SIZE, max_size=MAX_SIZE, threads=1, outdir=None, no_plot=True, verbose=False).res2d
    return pd.DataFrame({'set': res['Term'].to_numpy(), 'nes': res['NES'].astype(float).to_numpy(),
                         'fdr': res['FDR q-val'].astype(float).to_numpy()})


def gsea(data, types):
    """4: gseapy.prerank on each dataset's, type's and method's full score ranking."""
    rows = []
    for name, adata in data.items():
        for t in types:
            for m in rep.METHODS:
                r = _get_rank_genes_df(adata, t, m)[['names', 'scores']].astype({'scores': float})
                rows.append(prerank(r).assign(dataset=name, type=t, method=m))
    g = pd.concat(rows, ignore_index=True)[['dataset', 'type', 'method', 'set', 'nes', 'fdr']]
    return g.sort_values(['dataset', 'type', 'method', 'set'], ignore_index=True)


WELL_MEASURED = 0.25
EQUAL_K = (50, 20)
CEILING_K = 20        # the ceiling plot's genes per type and method


def confident(data, types):
    """Second diagnostic: readout 3 over one gene set shared by the three methods, per type: the genes significant
    (either direction) for every method in both datasets, and the genes detected in at least WELL_MEASURED of the cells
    of both the type and the rest in both datasets; and, per method, the genes that method calls significant in both."""
    rows, top_rows = [], []
    for t in types:
        res, sig, well = {}, {}, []
        for name, adata in data.items():
            own = (adata.obs['type'] == t).to_numpy()
            det = (adata.layers['norm_counts'] > 0).tocsc()
            p_in, p_out = np.asarray(det[own].mean(0)).ravel(), np.asarray(det[~own].mean(0)).ravel()
            well.append(set(adata.var_names[(p_in >= WELL_MEASURED) & (p_out >= WELL_MEASURED)]))
            for m in rep.METHODS:
                r = _get_rank_genes_df(adata, t, m).set_index('names')
                res[name, m] = r['scores'].astype(float)
                sig[name, m] = set(r.index[r['pvals_adj'] < 0.05])
        own = {m: sig['PBMC3k', m] & sig['PBMC10k', m] for m in rep.METHODS}
        for label, genes in (('confident', set.intersection(*sig.values())), ('well_measured', set.intersection(*well)),
                             ('confident_own', None)):
            for m in rep.METHODS:
                genes_m = own[m] if genes is None else genes
                both = pd.concat({name: res[name, m].reindex(sorted(genes_m)) for name in rep.DATASETS}, axis=1)
                both = both.replace([np.inf, -np.inf], np.nan).dropna()
                rows.append({'type': t, 'genes_set': label, 'method': m, 'genes': len(both),
                             'pearson': pearsonr(both['PBMC3k'], both['PBMC10k']).statistic if len(both) > 2 else np.nan,
                             'spearman': spearmanr(both['PBMC3k'], both['PBMC10k']).statistic if len(both) > 2 else np.nan})
        for k in EQUAL_K:
            for m in rep.METHODS:
                both = pd.concat({name: res[name, m].reindex(sorted(own[m])) for name in rep.DATASETS}, axis=1)
                both = both.replace([np.inf, -np.inf], np.nan).dropna()
                strength = pd.DataFrame({'gene': both.index, 'strength': both.abs().min(axis=1).to_numpy()})
                top = strength.sort_values(['strength', 'gene'], ascending=[False, True])['gene'].head(k)
                both = both.loc[top]
                if k == CEILING_K:
                    top_rows.append(both.rename(columns=lambda c: f'score_{c}').rename_axis('gene').reset_index()
                                    .assign(type=t, method=m))
                rows.append({'type': t, 'genes_set': f'confident_own_top{k}', 'method': m, 'genes': len(both),
                             'pearson': pearsonr(both['PBMC3k'], both['PBMC10k']).statistic,
                             'spearman': spearmanr(both['PBMC3k'], both['PBMC10k']).statistic})
    top = pd.concat(top_rows, ignore_index=True)[['type', 'method', 'gene', 'score_PBMC3k', 'score_PBMC10k']]
    return pd.DataFrame(rows), top


LFC_RANKINGS = {'ln_lfc': 'ln', 'scanpy_lfc': 't_test'}     # scanpy's logfoldchanges are the same for both baselines


def lfc(data, types):
    """Diagnostic: the same as 2 and 4 with genes ranked by fold change, over the genes detected in at least one cell
    of both the type and the rest, which leaves out the LFCs set by LN's zero-detection fallback or scanpy's
    pseudocount."""
    rankings, rows = {}, []
    for name, adata in data.items():
        det = (adata.layers['norm_counts'] > 0).tocsc()
        for t in types:
            own = (adata.obs['type'] == t).to_numpy()
            keep = adata.var_names[(np.asarray(det[own].sum(0)).ravel() > 0) & (np.asarray(det[~own].sum(0)).ravel() > 0)]
            for label, m in LFC_RANKINGS.items():
                r = _get_rank_genes_df(adata, t, m)[['names', 'logfoldchanges']].astype({'logfoldchanges': float})
                r = r[r['names'].isin(keep)]
                rankings[name, t, label] = r.set_index('names')['logfoldchanges']
                rows.append(prerank(r).assign(dataset=name, type=t, method=label))
    g = pd.concat(rows, ignore_index=True)[['dataset', 'type', 'method', 'set', 'nes', 'fdr']]
    g = g.sort_values(['dataset', 'type', 'method', 'set'], ignore_index=True)
    corr = []
    for t in types:
        for label in LFC_RANKINGS:
            both = pd.concat({name: rankings[name, t, label] for name in rep.DATASETS}, axis=1, join='inner').dropna()
            corr.append({'type': t, 'ranking': label, 'genes': len(both),
                         'ranking_spearman': spearmanr(both['PBMC3k'], both['PBMC10k']).statistic})
    return g, pd.DataFrame(corr)


def gsea_concordance(g):
    """Per method, pooled over the types and per type: the NES Spearman correlation between the datasets, and the
    agreement of the significant sets."""
    w = g.pivot_table(index=['type', 'method', 'set'], columns='dataset', values=['nes', 'fdr']).dropna()
    w.columns = [f'{v}_{d}' for v, d in w.columns]
    w = w.reset_index()
    rows = []
    for (m, t), d in [((m, 'pooled'), d) for m, d in w.groupby('method')] + list(w.groupby(['method', 'type'])):
        s3, s10 = d['fdr_PBMC3k'] < FDR, d['fdr_PBMC10k'] < FDR
        same = s3 & s10 & (np.sign(d['nes_PBMC3k']) == np.sign(d['nes_PBMC10k']))
        rows.append({'method': m, 'type': t, 'sets': len(d), 'nes_spearman': spearmanr(d['nes_PBMC3k'], d['nes_PBMC10k']).statistic,
                     'significant_pbmc3k': int(s3.sum()), 'significant_pbmc10k': int(s10.sum()),
                     'significant_both_same_sign': int(same.sum()),
                     'significant_agreement': same.sum() / max(int((s3 | s10).sum()), 1)})
    return w, pd.DataFrame(rows)


def summary(mk, sc, gc):
    """One row per method: each level's pooled concordance."""
    rows = []
    for m in rep.METHODS:
        s = sc[sc['method'] == m]
        g = gc[(gc['method'] == m) & (gc['type'] == 'pooled')].iloc[0]
        rows.append({'method': m, 'markers_top10': mk[(mk['method'] == m) & (mk['n'] == N_MAIN)]['concordance'].iloc[0],
                     'ranking_spearman': s['ranking_spearman'].mean(), 'top_spearman': s['top_spearman'].mean(),
                     'score_pearson': s['score_pearson'].mean(), 'score_slope': s['score_slope'].median(),
                     'nes_spearman': g['nes_spearman'], 'significant_agreement': g['significant_agreement']})
    return pd.DataFrame(rows)


LEVELS = {'markers_top10': f'Top-{N_MAIN} markers\n(overlap)', 'ranking_spearman': 'Full ranking\n(Spearman)',
          'top_spearman': f'Top-{TOP} union\n(Spearman)', 'score_pearson': 'Scores\n(Pearson)',
          'nes_spearman': 'GSEA NES\n(Spearman)', 'significant_agreement': 'Significant sets\n(agreement)'}


def draw_ceiling(ax, top, conf, methods=rep.METHODS):
    """Each method's CEILING_K strongest genes confident in both datasets, per type: the score in PBMC3k against
    PBMC10k, each divided by the largest of its type's, method's and dataset's; the mean over types of the Pearson
    correlation of the raw scores, per method."""
    r = conf[conf['genes_set'] == f'confident_own_top{CEILING_K}'].groupby('method')['pearson'].mean()
    for m in method_draw_order(methods):
        d = top[top['method'] == m]
        scale = lambda c: d[c].abs() / d.groupby('type')[c].transform(lambda v: v.abs().max())
        ax.scatter(scale('score_PBMC10k'), scale('score_PBMC3k'), s=5, lw=0, alpha=0.7, color=method_color(m),
                   label=method_label(m))
    for k, m in enumerate(methods):
        ax.text(0.97, 0.03 + 0.09 * (len(methods) - 1 - k), f'{method_label(m)}: r = {r[m]:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=plot_style.SMALL, color=method_color(m))
    ax.plot([0, 1], [0, 1], color='0.6', lw=0.6, ls='--')
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('PBMC10k score / strongest')
    ax.set_ylabel('PBMC3k score / strongest')


def plot(summ, nes, top, conf):
    """The concordance at each level per method, each method's NES in PBMC3k against PBMC10k, and the ceiling."""
    plot_style.use()
    fig, ax = plt.subplots(figsize=plot_style.figsize(2))
    x = np.arange(len(LEVELS))
    width = 0.8 / len(rep.METHODS)
    for i, m in enumerate(rep.METHODS):
        y = summ.set_index('method').loc[m, list(LEVELS)]
        ax.bar(x + (i - 1) * width, y, width, color=method_color(m), label=method_label(m), edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(list(LEVELS.values()))
    ax.set_ylabel('Concordance, PBMC3k and PBMC10k')
    ax.set_ylim(0, 1)
    rep.save(fig, OUT / fig_name('concordance_levels'))

    fig, axes = plt.subplots(1, len(rep.METHODS), figsize=plot_style.figsize(len(rep.METHODS)), sharex=True, sharey=True)
    for ax, m in zip(axes, rep.METHODS):
        d = nes[nes['method'] == m]
        ax.scatter(d['nes_PBMC3k'], d['nes_PBMC10k'], s=5, color=method_color(m), linewidths=0)
        ax.axhline(0, color='0.6', lw=0.6)
        ax.axvline(0, color='0.6', lw=0.6)
        ax.set_title(method_label(m))
        ax.set_xlabel('NES, PBMC3k')
    axes[0].set_ylabel('NES, PBMC10k')
    rep.save(fig, OUT / fig_name('gsea_nes'), key=False)

    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_ceiling(ax, top, conf)
    rep.save(fig, OUT / fig_name('score_ceiling'))


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset concordance of markers, rankings, scores and GSEA')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('markers', 'scores', 'gsea', 'gsea_pairs', 'gsea_concordance', 'summary',
                                          'lfc_rankings', 'lfc_gsea', 'lfc_gsea_concordance', 'confident_scores',
                                          'confident_top')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the concordance {k}', source='this script without --plot-only'))
                  for k, p in csv.items()}
    else:
        require_input(GMT, what='the MSigDB Hallmark 2020 gene sets', source="gseapy.get_library('MSigDB_Hallmark_2020')")
        data, types, sizes = tests()
        mk = markers(data, types)
        sc = scores(data, types, sizes)
        g = gsea(data, types)
        pairs, gc = gsea_concordance(g)
        lg, lr = lfc(data, types)
        conf, conf_top = confident(data, types)
        tables = {'markers': mk, 'scores': sc, 'gsea': g, 'gsea_pairs': pairs, 'gsea_concordance': gc,
                  'summary': summary(mk, sc, gc), 'lfc_rankings': lr, 'lfc_gsea': lg,
                  'lfc_gsea_concordance': gsea_concordance(lg)[1], 'confident_scores': conf, 'confident_top': conf_top}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['summary'], tables['gsea_pairs'], tables['confident_top'], tables['confident_scores'])


if __name__ == '__main__':
    main()
