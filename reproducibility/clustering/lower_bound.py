#!/usr/bin/env python3
"""Do LN's markers, ranked by the lower confidence bound of their fold change, replicate and read better than by score?

Each rule orders a method's Bonferroni-significant genes with a positive score: by the score (the practice rule), by the
reported LFC, and, for LN, by LFC - Z * SE (SE = LFC / score, exact for LN) over the genes detected in the comparison
group (genes it lacks get their LFC and SE from LN's zero-detection fallback); as sensitivities, the same with those
genes kept, and with the Bonferroni quantile of the comparison's genes for Z; and, for the log1p t-test in readout A, the
lower bound of its own effect, the difference in mean log1p d, as d - Z d / t. Readouts: A, clustering/replication.py's
cross-dataset concordance; B, curated-marker recovery in each Leiden cluster's top TOP, as clustering/metrics.py computes
it; C, the share of ribosomal and mitochondrial genes in each top TOP; D, in the PBMC10k subclusters of
clustering/subcluster_de.py that another lineage's protein separates from their siblings, the share of each top TOP
that are that lineage's genes. Every top TOP also records its share of genes undetected in the comparison group.

    python -m clustering.lower_bound               # from reproducibility/, minutes
    python -m clustering.lower_bound --plot-only   # redraw from the CSVs
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
from scipy.stats import norm, rankdata

import plot_style
from clustering import replication as rep
from clustering import subcluster_de as sub
from clustering.celltype_labels import RAW, dataset_paths
from clustering.heldout_fp import de_output
from evaluation_utils import _get_rank_genes_df, get_top_sig_genes_adata, recall
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

Z = 1.96
TOP = 20
MIN_SUB = 100            # subclusters read in C and D from this many cells
AUC_MIN = 0.9
PARTNER_FOLD, PARTNER_MIN = 4.0, 1.0
LINEAGES = {'T': (('Naive CD4 T', 'Memory CD4 T', 'CD8 T'), 'CD3_TotalC'), 'B': (('B',), 'CD19_TotalC'),
            'monocyte': (('CD14 Mono', 'FCGR3A Mono'), 'CD14_TotalC')}
RULES = {'ln': ('score', 'lfc', 'lower_bound', 'lower_bound_all', 'lower_bound_bonf'),
         't_test': ('score', 'lfc', 'log1p_bound'), 'wilcoxon': ('score', 'lfc')}
EXCLUDES_ABSENT = ('lower_bound', 'lower_bound_bonf')
TECHNICAL = r'^(?:RPL\d+[A-Z]*\d*|RPS\d+[A-Z]?\d*|RPSA|RPLP[0-2]|MT-.+)$'
STYLE = {'score': '-', 'lfc': ':', 'lower_bound': '--', 'lower_bound_all': (0, (1, 1)), 'lower_bound_bonf': '-.',
         'log1p_bound': '--'}
RULE_LABEL = {'score': 'score', 'lfc': 'LFC', 'lower_bound': 'lower bound', 'lower_bound_all': 'lower bound, all genes',
              'lower_bound_bonf': 'lower bound, Bonferroni z', 'log1p_bound': 'own lower bound'}
OUT = results_dir(__file__) / 'lower_bound'


def ordered(res, rule, universe=None, d=None, genes=None, absent=None):
    """A method's significant genes with a positive score, best first under rule; ties broken by score, then name.
    absent marks the genes undetected in the comparison group; the Bonferroni quantile counts the genes tested, all of
    res's rows unless given."""
    if rule in ('score', 'lfc'):
        return rep.markers(res, res['names'] if universe is None else universe, rule)
    s = res[(res['pvals_adj'] < 0.05) & (res['scores'] > 0)]
    if universe is not None:
        s = s[s['names'].isin(universe)]
    if rule in EXCLUDES_ABSENT:
        s = s[~absent.reindex(s['names']).to_numpy(dtype=bool)]
    if rule == 'log1p_bound':
        effect = d.reindex(s['names']).to_numpy()
    else:
        effect = s['logfoldchanges'].to_numpy(dtype=float)
    z = norm.isf(0.05 / (2 * (genes or len(res)))) if rule == 'lower_bound_bonf' else Z
    s = s.assign(bound=effect - z * effect / s['scores'].to_numpy(dtype=float))
    return s.sort_values(['bound', 'scores', 'names'], ascending=[False, False, True])


def technical_share(genes):
    return float(pd.Series(list(genes), dtype=str).str.match(TECHNICAL).mean()) if len(genes) else np.nan


def replication():
    """A: each dataset's, rule's, method's and type's top markers, and their concordance; the score and LFC rules must
    reproduce clustering/replication.py's markers and pooled concordance."""
    data = {name: rep.load(name) for name in rep.DATASETS}
    for adata in data.values():
        rep.run_tests(adata)
    types = sorted(data['PBMC3k'].obs['type'].cat.categories)
    universe = {'PBMC3k': rep.annotation('PBMC10k'), 'PBMC10k': rep.annotation('PBMC3k')}
    rows = []
    for name, adata in data.items():
        detected = (adata.layers['norm_counts'] > 0).tocsc()
        log1p = adata.layers['log1p_norm']
        for t in types:
            own = (adata.obs['type'] == t).to_numpy()
            absent = pd.Series(np.asarray(detected[~own].sum(0)).ravel() == 0, index=adata.var_names)
            d = pd.Series(np.asarray(log1p[own].mean(0)).ravel() - np.asarray(log1p[~own].mean(0)).ravel(),
                          index=adata.var_names)
            for m, rules in RULES.items():
                res = _get_rank_genes_df(adata, t, m)
                for rule in rules:
                    top = ordered(res, rule, universe[name], d, absent=absent).head(max(rep.NS))
                    rows += [{'dataset': name, 'ranking': rule, 'method': m, 'type': t, 'rank': k + 1, 'gene': r.names,
                              'logfoldchanges': r.logfoldchanges, 'scores': r.scores,
                              'undetected_in_rest': bool(absent[r.names])} for k, r in enumerate(top.itertuples())]
    table = pd.DataFrame(rows)
    cols = ['dataset', 'ranking', 'method', 'type', 'rank', 'gene']
    check = pd.read_csv(require_input(rep.OUT / 'markers.csv', what='the replication markers',
                                      source='python -m clustering.replication'))
    mine = table[table['ranking'].isin(rep.RANKINGS)][cols].sort_values(cols[:5]).reset_index(drop=True)
    if not mine.equals(check[cols].sort_values(cols[:5]).reset_index(drop=True)):
        raise SystemExit('the score and LFC rules do not reproduce replication/markers.csv; nothing written')
    conc = rep.concordance(table)
    pool = rep.pooled(conc)
    keys = ['ranking', 'method', 'n']
    ref = pd.read_csv(rep.OUT / 'concordance_pooled.csv').sort_values(keys).reset_index(drop=True)
    got = pool[pool['ranking'].isin(rep.RANKINGS)].sort_values(keys).reset_index(drop=True)
    if not (got[keys].astype(str).equals(ref[keys].astype(str))
            and np.allclose(got['concordance'], ref['concordance'], rtol=0, atol=1e-12)):
        raise SystemExit('the score and LFC rules do not reproduce replication/concordance_pooled.csv; nothing written')
    return table, conc, pool


def de_light(path, keys):
    """A DE output's obs, its genes' detection, and the rank_genes_groups results under keys."""
    with h5py.File(path, 'r') as f:
        adata = ad.AnnData(obs=ad.io.read_elem(f['obs']), var=ad.io.read_elem(f['var']))
        detected = (sp.csr_matrix(ad.io.read_elem(f['layers/counts'])) > 0).tocsc()
        for k in keys:
            adata.uns[k] = ad.io.read_elem(f[f'uns/{k}'])
    return adata, detected


def recovery():
    """B: per dataset, resolution, method and rule, each curated type's best recall over the clusters' top TOP, and the
    mean share of the clusters' top TOP undetected in the rest; the score rule must reproduce clustering/metrics.py's
    best_match_top20_by_resolution.csv."""
    labels = {'ln': 'LN test', 't_test': 'Scanpy t-test (log1p)', 'wilcoxon': 'Scanpy Wilcoxon (log1p)'}
    rows = []
    for name, (config_path, out) in dataset_paths().items():
        config = yaml.safe_load(open(config_path))
        keys = [k for _, k in config['leiden_resolutions']]
        adata, detected = de_light(require_input(de_output(config['adata_path'], str(out)), what=f'the {name} DE output',
                                                 source='python -m clustering.de'), [f'{m}_{k}' for m in RULES for k in keys])
        total = np.asarray(detected.sum(0)).ravel()
        curated = {ct: set(g) for ct, g in config['markers'].items()}
        for key in keys:
            groups = adata.obs[key].astype(str).to_numpy()
            clusters = sorted(np.unique(groups), key=lambda v: int(v) if v.isdigit() else v)
            absent = {c: pd.Series(total - np.asarray(detected[groups == c].sum(0)).ravel() == 0, index=adata.var_names)
                      for c in clusters}
            for m, rules in RULES.items():
                for rule in (r for r in rules if r != 'log1p_bound'):
                    tops = {}
                    for c in clusters:
                        if rule == 'score':
                            tops[c] = get_top_sig_genes_adata(adata, c, top_n=TOP, key=f'{m}_{key}', pval_threshold=0.05,
                                                              positive_score_only=True)
                        else:
                            res = _get_rank_genes_df(adata, c, f'{m}_{key}')
                            tops[c] = set(ordered(res, rule, absent=absent[c])['names'].head(TOP))
                    undetected = np.nanmean([np.mean([absent[c][g] for g in tops[c]]) if tops[c] else np.nan
                                             for c in clusters])
                    for ct, genes in curated.items():
                        rows.append({'dataset': name, 'resolution': key, 'clusters': len(clusters), 'method': m,
                                     'method_label': labels[m], 'rule': rule, 'cell_type': ct,
                                     'best_recall': max(recall(tops[c], genes) for c in clusters),
                                     'undetected': undetected})
        check = pd.read_csv(require_input(out / 'metrics' / 'best_match_top20_by_resolution.csv',
                                          what=f'the {name} curated-marker recovery', source='python -m clustering.metrics'))
        mine = pd.DataFrame(rows)
        mine = mine[(mine['dataset'] == name) & (mine['rule'] == 'score')]
        merged = check.merge(mine, left_on=['resolution', 'method', 'cell_type'],
                             right_on=['resolution', 'method_label', 'cell_type'])
        if len(merged) != len(check) or not np.allclose(merged['best_recall_x'], merged['best_recall_y']):
            raise SystemExit(f'the score rule does not reproduce {name} best_match_top20_by_resolution.csv; nothing written')
    return pd.DataFrame(rows).drop(columns='method_label')


def adt(cells):
    """Raw counts of the lineage proteins for the PBMC10k cells."""
    full = sc.read_10x_h5(str(require_input(RAW, what="the 5' PBMC 10k matrix", source='python -m fetch_data --arm citeseq')),
                          gex_only=False)
    full.var_names_make_unique()
    proteins = [p for _, p in LINEAGES.values()]
    return pd.DataFrame(full[:, proteins].X.toarray(), index=full.obs_names, columns=proteins).loc[cells]


def auc(x, own):
    """P(a subcluster cell's value exceeds a sibling's), ties counted one half."""
    r = rankdata(x)
    n1, n2 = own.sum(), (~own).sum()
    return (r[own].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def lineage_of(t):
    return next((k for k, (types, _) in LINEAGES.items() if t in types), None)


def subclusters():
    """C and D over the stage 3d subclusters of at least MIN_SUB cells: each method's and rule's top TOP from
    subclustering/markers.csv, what it is made of, and, in PBMC10k subclusters another lineage's protein separates,
    its share of that lineage's genes. Memberships are recomputed as clustering/subcluster_de.py computes them and must
    reproduce its subclusters.csv."""
    marks = pd.read_csv(require_input(sub.OUT / 'markers.csv', what='the subclustering markers',
                                      source='python -m clustering.subcluster_de'))
    subs = pd.read_csv(sub.OUT / 'subclusters.csv')
    for t in (subs, marks):
        t['subcluster'] = t['subcluster'].astype(str)
    rows = []
    for name in rep.DATASETS:
        adata = rep.load(name)
        adata.layers['counts'] = sub.counts(name, adata.obs_names)
        x = adata.layers['norm_counts'].tocsr()
        means = {t: np.asarray(x[(adata.obs['type'] == t).to_numpy()].mean(0)).ravel()
                 for t in adata.obs['type'].cat.categories}
        lineage_means = {k: np.asarray(x[adata.obs['type'].isin(types).to_numpy()].mean(0)).ravel()
                         for k, (types, _) in LINEAGES.items()}
        protein = adt(adata.obs_names) if name == 'PBMC10k' else None
        sizes = adata.obs['type'].value_counts()
        for t in sorted(sizes[sizes >= sub.MIN_CELLS].index):
            parent = adata[(adata.obs['type'] == t).to_numpy()].copy()
            detected = (parent.layers['counts'] > 0).tocsc()
            for r, labels in sub.subcluster(parent).items():
                if len(np.unique(labels)) < 2:
                    continue
                for c in np.unique(labels):
                    own = labels == c
                    key = {'dataset': name, 'type': t, 'resolution': r, 'subcluster': c}
                    expect = subs[(subs['dataset'] == name) & (subs['type'] == t) & (subs['resolution'] == r)
                                  & (subs['subcluster'] == c)]
                    if len(expect) != 1 or int(expect['n_cells'].iloc[0]) != int(own.sum()):
                        raise SystemExit(f'subclusters do not reproduce subclusters.csv at {key}; nothing written')
                    if own.sum() < MIN_SUB:
                        continue
                    absent = pd.Series(np.asarray(detected[~own].sum(0)).ravel() == 0, index=parent.var_names)
                    partner, protein_auc = None, np.nan
                    if protein is not None:
                        aucs = {k: auc(protein.loc[parent.obs_names, p].to_numpy(), own) for k, (_, p) in LINEAGES.items()
                                if k != lineage_of(t)}
                        k, a = max(aucs.items(), key=lambda kv: kv[1])
                        if a >= AUC_MIN:
                            partner, protein_auc = k, a
                    partner_genes = set()
                    if partner:
                        keep = (lineage_means[partner] >= PARTNER_FOLD * means[t]) & (lineage_means[partner] >= PARTNER_MIN)
                        partner_genes = set(adata.var_names[keep])
                    m_c = marks[(marks['dataset'] == name) & (marks['type'] == t) & (marks['resolution'] == r)
                                & (marks['subcluster'] == c)]
                    genes_tested = int(expect['genes_tested'].iloc[0])
                    for m, rules in RULES.items():
                        res = m_c[m_c['method'] == m].rename(columns={'gene': 'names'})
                        for rule in (r_ for r_ in rules if r_ != 'log1p_bound'):
                            top = list(ordered(res, rule, genes=genes_tested, absent=absent)['names'].head(TOP))
                            rows.append({**key, 'n_cells': int(own.sum()), 'depth_ratio': expect['depth_ratio'].iloc[0],
                                         'method': m, 'rule': rule, 'n_top': len(top),
                                         'technical_share': technical_share(top),
                                         'undetected_in_siblings': float(absent[top].mean()) if top else np.nan,
                                         'partner': partner, 'partner_protein_auc': protein_auc,
                                         'partner_share': float(np.mean([g in partner_genes for g in top]))
                                         if partner and top else np.nan})
    return pd.DataFrame(rows)


def composition(table, subs):
    """C: per source, method and rule, the mean share of ribosomal and mitochondrial genes in the top TOP and of genes
    undetected in the comparison group, per dataset and pooled over every top TOP of both sources (prediction 4)."""
    t = table[(table['rank'] <= TOP) & (table['ranking'] != 'log1p_bound')]
    types = t.groupby(['dataset', 'ranking', 'method', 'type']).agg(
        technical_share=('gene', technical_share), undetected=('undetected_in_rest', 'mean')).reset_index()
    types = types.rename(columns={'ranking': 'rule'}).assign(source='types')
    subs = subs.rename(columns={'undetected_in_siblings': 'undetected'}).assign(source='subclusters')
    lists = pd.concat([types, subs], ignore_index=True)[['source', 'dataset', 'method', 'rule', 'technical_share',
                                                        'undetected']]
    per = lists.groupby(['source', 'dataset', 'method', 'rule'])[['technical_share', 'undetected']].mean().reset_index()
    pooled = lists.groupby(['method', 'rule'])[['technical_share', 'undetected']].mean().reset_index()
    return pd.concat([per, pooled.assign(source='pooled', dataset='both')], ignore_index=True)


def doublet_summary(subs):
    """D: per method and rule, the mean share of partner-lineage genes over the subclusters with a partner."""
    d = subs[subs['partner'].notna()]
    if d.empty:
        print(f'  readout D: no PBMC10k subcluster of at least {MIN_SUB} cells has another lineage\'s protein at AUC >= '
              f'{AUC_MIN}')
        return pd.DataFrame(columns=['method', 'rule', 'subclusters', 'partner_share'])
    return d.groupby(['method', 'rule']).agg(subclusters=('partner_share', 'size'),
                                             partner_share=('partner_share', 'mean')).reset_index()


def lines(ax, d, x, y):
    """One line per method and rule, LN drawn last."""
    for m in method_draw_order(d['method'].unique()):
        for rule in RULES[m]:
            e = d[(d['method'] == m) & (d['rule'] == rule)].sort_values(x)
            if len(e):
                ax.plot(e[x], e[y], ls=STYLE[rule], color=method_color(m), marker='o', ms=3,
                        label=f'{method_label(m)}, {RULE_LABEL[rule]}')


def bars(ax, d, column, title):
    """Horizontal bars per method and rule."""
    d = d.assign(order=d['method'].map({m: i for i, m in enumerate(RULES)}),
                 rorder=[RULES[m].index(r) for m, r in zip(d['method'], d['rule'])]).sort_values(['order', 'rorder'])
    y = np.arange(len(d))[::-1]
    ax.barh(y, d[column], color=[method_color(m) for m in d['method']], edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{method_label(m)}, {RULE_LABEL[r]}' for m, r in zip(d['method'], d['rule'])])
    ax.set_title(title)


def plot(pool, rec, comp, dsum):
    """A: concordance against N; B: curated recovery against the number of clusters; C: the top TOP's ribosomal and
    mitochondrial share; D: the partner-lineage share, if any subcluster has a partner."""
    plot_style.use()
    fig, ax = plt.subplots(figsize=plot_style.figsize(1))
    lines(ax, pool.rename(columns={'ranking': 'rule'}), 'n', 'concordance')
    ax.set_xlabel('Top N markers')
    ax.set_ylabel('Concordance, PBMC3k and PBMC10k')
    ax.set_ylim(bottom=0)
    rep.save(fig, OUT / fig_name('concordance_by_rule'))

    fig, axes = plt.subplots(1, len(rep.DATASETS), figsize=plot_style.figsize(len(rep.DATASETS)))
    mean = rec.groupby(['dataset', 'clusters', 'method', 'rule'])['best_recall'].mean().reset_index()
    for ax, name in zip(axes, rep.DATASETS):
        lines(ax, mean[mean['dataset'] == name], 'clusters', 'best_recall')
        ax.set_xscale('log')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel(f'Curated markers in the top {TOP}')
        ax.set_ylim(0, 1)
        ax.set_title(name)
    rep.save(fig, OUT / fig_name('recovery_by_rule'))

    d = comp[comp['source'] != 'pooled'].groupby(['source', 'method', 'rule'])['technical_share'].mean().reset_index()
    n = d.groupby('source').size().max()
    fig, axes = plt.subplots(1, 2, figsize=(2 * plot_style.PANEL[0] + 1.6, 0.16 * n + 0.8), sharey=True)
    for ax, source in zip(axes, ('types', 'subclusters')):
        bars(ax, d[d['source'] == source], 'technical_share',
             'cell types' if source == 'types' else f'subclusters of at least {MIN_SUB} cells')
        ax.set_xlabel(f'Ribosomal and MT genes\nin the top {TOP}')
    rep.save(fig, OUT / fig_name('composition_by_rule'), key=False)

    if len(dsum):
        fig, ax = plt.subplots(figsize=(plot_style.PANEL[0] + 1.6, 0.16 * len(dsum) + 0.8))
        bars(ax, dsum, 'partner_share', 'PBMC10k subclusters with another lineage\'s protein')
        ax.set_xlabel(f'Partner-lineage genes in the top {TOP}')
        rep.save(fig, OUT / fig_name('doublet_partner_by_rule'), key=False)


def main():
    parser = argparse.ArgumentParser(description='LN markers ranked by the lower bound of the fold change, against the score')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('markers', 'concordance', 'concordance_pooled', 'recovery', 'subclusters',
                                          'composition', 'doublet_partner')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the lower-bound {k}', source='this script without --plot-only'))
                  for k, p in csv.items()}
    else:
        table, conc, pool = replication()
        rec = recovery()
        subs = subclusters()
        tables = {'markers': table, 'concordance': conc, 'concordance_pooled': pool, 'recovery': rec,
                  'subclusters': subs, 'composition': composition(table, subs), 'doublet_partner': doublet_summary(subs)}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['concordance_pooled'], tables['recovery'], tables['composition'], tables['doublet_partner'])


if __name__ == '__main__':
    main()
