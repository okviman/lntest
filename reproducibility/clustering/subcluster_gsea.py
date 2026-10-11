#!/usr/bin/env python3
"""Subcluster states read through GSEA: which of each method's pathway calls survive matching the two sides' depth?

The splits are clustering/subcluster_recursive.py's, made again the same way. Each child against its siblings, when both
have at least MIN_CELLS cells (a two-way split once, through its first child), is tested with the three tests as
clustering/de.py runs them, and each method's full score ranking goes through gseapy prerank on the Hallmark sets as
clustering/concordance.py runs it. Then the deeper side's counts are binomially thinned by the ratio of the two sides'
median depths, CP10K and log1p CP10K are recomputed, and tests and prerank rerun, once per seed in SEEDS. A set is
called at FDR q < FDR; after matching, when called with the same sign under most seeds. Per comparison and method: the
sets called as observed and after matching, the share of observed calls that survive, the share whose NES has the sign
of the depth gap, and the top TOP markers' overlap with LN's. Comparisons whose depth ratio is at least GAP either way
are pooled apart from the others.

    python -m clustering.subcluster_gsea               # from reproducibility/, about 10 minutes
    python -m clustering.subcluster_gsea --plot-only   # redraw from the CSVs
"""

import argparse

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

import plot_style
from clustering.concordance import GMT, MAX_SIZE, MIN_SIZE, PERMUTATIONS, SEED
from clustering.replication import DATASETS, METHODS, load, run_tests, save
from clustering.subcluster_de import MIN_DETECTED, counts
from clustering.subcluster_recursive import MAX_LEVEL, MIN_PARENT, MIN_TYPE, split
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

MIN_CELLS = 100
GAP = 1.5
FDR = (0.25, 0.05)       # the first is the call; the second is reported alongside
SEEDS = (0, 1, 2)
TOP = 20
OUT = results_dir(__file__) / 'subclustering' / 'gsea'


def comparisons(name, adata):
    """The dataset's splits, breadth first as clustering/subcluster_recursive.py makes them; per kept comparison, the
    parent's cells and the child's mask within them."""
    sizes = adata.obs['type'].value_counts()
    queue = [(t, np.flatnonzero((adata.obs['type'] == t).to_numpy()), 1, '')
             for t in sorted(sizes[sizes >= MIN_TYPE].index)]
    while queue:
        t, cells, level, path = queue.pop(0)
        if level > MAX_LEVEL or len(cells) < MIN_PARENT:
            continue
        r, labels = split(adata[cells].copy())
        if labels is None:
            continue
        children = sorted(np.unique(labels), key=lambda v: int(v))
        for k, c in enumerate(children):
            own = labels == c
            if own.sum() >= MIN_CELLS and (~own).sum() >= MIN_CELLS and not (len(children) == 2 and k == 1):
                yield {'dataset': name, 'type': t, 'level': level, 'path': path + c, 'resolution': r}, cells, own
            queue.append((t, cells[own], level + 1, path + c + '.'))


def tested(x, own, keep, genes):
    """The child against its siblings from counts x (cells by the arm's genes): CP10K and log1p CP10K over all genes,
    the three tests over the genes at positions keep, named genes; the child's result per method."""
    total = np.asarray(x.sum(1)).ravel()
    cp = sp.csr_matrix(sp.diags(1e4 / total) @ x, dtype=np.float32)[:, keep]
    lg = cp.copy()
    lg.data = np.log1p(lg.data)
    a = ad.AnnData(X=cp, obs=pd.DataFrame({'side': pd.Categorical(np.where(own, 'child', 'siblings'))},
                                          index=np.arange(len(own)).astype(str)),
                   var=pd.DataFrame(index=genes))
    a.layers['norm_counts'], a.layers['log1p_norm'] = cp, lg
    run_tests(a, 'side')
    return {m: _get_rank_genes_df(a, 'child', m) for m in METHODS}


def prerank(res):
    """gseapy prerank on a method's full score ranking; each Hallmark set's NES and FDR q."""
    r = res[['names', 'scores']].astype({'scores': float})
    r = r[np.isfinite(r['scores'])].sort_values(['scores', 'names'], ascending=[False, True]).reset_index(drop=True)
    g = gp.prerank(rnk=r, gene_sets=str(GMT), permutation_num=PERMUTATIONS, seed=SEED, min_size=MIN_SIZE,
                   max_size=MAX_SIZE, threads=4, outdir=None, no_plot=True, verbose=False).res2d
    return pd.DataFrame({'set': g['Term'].to_numpy(), 'nes': g['NES'].astype(float).to_numpy(),
                         'fdr': g['FDR q-val'].astype(float).to_numpy()})


def top_markers(res):
    """The child's top TOP genes by score."""
    r = res[np.isfinite(res['scores'].astype(float))]
    return set(r.sort_values(['scores', 'names'], ascending=[False, True])['names'].head(TOP))


def run():
    """Every comparison's sets as observed and after matching per seed, and its descriptors."""
    rows, sets = [], []
    for name in DATASETS:
        adata = load(name)
        adata.layers['counts'] = counts(name, adata.obs_names)
        x_all = sp.csr_matrix(adata.layers['counts'], dtype=np.float64)
        for key, cells, own in comparisons(name, adata):
            x = x_all[cells]
            keep = np.flatnonzero(np.asarray((x > 0).sum(0)).ravel() >= MIN_DETECTED)
            genes = adata.var_names[keep]
            depth = np.asarray(x.sum(1)).ravel()
            ratio = np.median(depth[own]) / np.median(depth[~own])
            deep = own if ratio > 1 else ~own
            factor = min(ratio, 1 / ratio)
            res = tested(x, own, keep, genes)
            tops = {m: top_markers(r) for m, r in res.items()}
            rows.append({**key, 'cells': int(own.sum()), 'siblings': int((~own).sum()), 'depth_ratio': ratio,
                         'gapped': bool(ratio >= GAP or ratio <= 1 / GAP), 'thinned': 'child' if ratio > 1 else 'siblings',
                         'factor': factor, **{f'top{TOP}_shared_ln_{m}': len(tops[m] & tops['ln']) / TOP
                                              for m in METHODS if m != 'ln'}})
            for m, r in res.items():
                sets.append(prerank(r).assign(**key, method=m, arm='observed'))
            for s in SEEDS:
                rng = np.random.default_rng(s)
                xt = x.copy()
                for i in np.flatnonzero(deep):
                    a_, b_ = xt.indptr[i], xt.indptr[i + 1]
                    xt.data[a_:b_] = rng.binomial(xt.data[a_:b_].astype(np.int64), factor)
                xt.eliminate_zeros()
                for m, r in tested(xt, own, keep, genes).items():
                    sets.append(prerank(r).assign(**key, method=m, arm=f'seed{s}'))
            print(f'  {name} {key["type"]} {key["path"]}: {own.sum()} against {(~own).sum()}, depth ratio {ratio:.2f}',
                  flush=True)
    return pd.DataFrame(rows), pd.concat(sets, ignore_index=True)


def calls(comp, sets):
    """Per comparison, method and threshold: observed calls, calls after matching, survivals, calls with the sign of
    the depth gap, and calls only after matching."""
    rows = []
    key = ['dataset', 'type', 'path']
    for (d, t, p, m), g in sets.groupby(key + ['method']):
        c = comp[(comp['dataset'] == d) & (comp['type'] == t) & (comp['path'] == p)].iloc[0]
        obs = g[g['arm'] == 'observed'].set_index('set')
        matched = g[g['arm'] != 'observed']
        for q in FDR:
            called_obs = obs.index[obs['fdr'] < q]
            sign_obs = np.sign(obs['nes'])
            m_called = matched[matched['fdr'] < q].assign(sign=lambda v: np.sign(v['nes']))
            votes = m_called.groupby(['set', 'sign']).size()
            after = {s: sg for (s, sg), v in votes.items() if v > len(SEEDS) / 2}
            survived = [s for s in called_obs if after.get(s) == sign_obs[s]]
            gap_sign = np.sign(np.log(c['depth_ratio']))
            rows.append({'dataset': d, 'type': t, 'path': p, 'level': c['level'], 'gapped': c['gapped'],
                         'depth_ratio': c['depth_ratio'], 'method': m, 'fdr': q, 'observed': len(called_obs),
                         'matched': len(after), 'survived': len(survived),
                         'with_gap_sign': int((sign_obs[called_obs] == gap_sign).sum()),
                         'only_matched': len(set(after) - set(called_obs)),
                         'observed_sets': ';'.join(sorted(called_obs)), 'matched_sets': ';'.join(sorted(after))})
    return pd.DataFrame(rows).sort_values(key + ['method', 'fdr'], ignore_index=True)


def summarise(per):
    """Per dataset (and both pooled), depth gap, method and threshold: comparisons and pooled shares."""
    out = []
    for subset, d in [('pooled', per), *per.groupby('dataset')]:
        g = d.groupby(['gapped', 'method', 'fdr'])
        s = pd.DataFrame({'comparisons': g.size(), 'observed': g['observed'].sum(), 'survived': g['survived'].sum(),
                          'with_gap_sign': g['with_gap_sign'].sum(), 'only_matched': g['only_matched'].sum()})
        s['survival'] = s['survived'] / s['observed']
        s['gap_sign_share'] = s['with_gap_sign'] / s['observed']
        out.append(s.reset_index().assign(dataset=subset))
    return pd.concat(out, ignore_index=True)


def draw_nes(ax, sets, comp, method):
    """Over depth-gapped comparisons, each of a method's observed calls: its NES as observed against its mean NES with
    depth matched."""
    key = ['dataset', 'type', 'path']
    s = sets.merge(comp.loc[comp['gapped'].astype(bool), key], on=key)
    g = s[s['method'] == method]
    obs = g[g['arm'] == 'observed'].set_index(key + ['set'])
    mat = g[g['arm'] != 'observed'].groupby(key + ['set'])['nes'].mean()
    sel = obs[obs['fdr'] < FDR[0]]
    lim = [-3, 3]
    ax.scatter(sel['nes'], mat.reindex(sel.index), s=6, lw=0, color=method_color(method))
    ax.plot(lim, lim, color='0.4', lw=0.8, ls='--')
    ax.axhline(0, color='0.8', lw=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('NES as observed')
    ax.set_ylabel('NES, depth matched')
    ax.set_title(method_label(method))


def draw_survival(ax, summary, methods=METHODS):
    """The share of observed calls surviving depth matching, pooled over both datasets, without and with a depth gap."""
    d = summary[(summary['dataset'] == 'pooled') & (summary['fdr'] == FDR[0])]
    for k, m in enumerate(method_draw_order(methods)):
        e = d[d['method'] == m].set_index('gapped').reindex([False, True])
        ax.plot(np.arange(2) + (k - 1) * 0.12, e['survival'], ls='none', marker='o', ms=plot_style.ERRORBAR['ms'],
                color=method_color(m), label=method_label(m))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Depth within\n{GAP}×', f'Depth gap\n≥ {GAP}×'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Pathway calls surviving\ndepth matching')


def plot(summary, sets, comp):
    """The arm's figures, one plot each: per method, the depth-gapped calls' NES as observed against depth matched;
    and the survival of calls without and with a depth gap."""
    plot_style.use()
    for m in METHODS:
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_nes(ax, sets, comp, m)
        save(fig, OUT / fig_name(f'subcluster_gsea_nes_{m}'), key=False)
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_survival(ax, summary)
    save(fig, OUT / fig_name('subcluster_gsea_survival'))


def main():
    parser = argparse.ArgumentParser(description='Subcluster pathway calls, as observed and depth matched')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('comparisons', 'sets', 'calls', 'summary')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the subcluster GSEA {k}', source='this script without --plot-only'),
                              dtype={'path': str}) for k, p in csv.items()}
    else:
        comp, sets = run()
        per = calls(comp, sets)
        tables = {'comparisons': comp, 'sets': sets, 'calls': per, 'summary': summarise(per)}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['summary'], tables['sets'], tables['comparisons'])


if __name__ == '__main__':
    main()
