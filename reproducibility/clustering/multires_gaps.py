#!/usr/bin/env python3
"""Does clustering at finer Leiden resolution open variance gaps between each cluster and the rest, and how do each
method's calls, top markers and pathway enrichments behave across resolutions, judged against sorted-cell bulk?

Reads clustering/de.py's outputs (each cluster against all other cells, five resolutions, both datasets) and reruns no
test. Per comparison, three gaps between the cluster and the rest, on CP10K over the genes detected in at least
MIN_DETECT cells on each side: the median |log2 variance ratio| over the genes whose means agree within EQUAL_MEAN
(log2), the same over all those genes, and |log2| of the ratio of median counts per cell. Per comparison and method: the
number of Bonferroni-significant genes (either direction, and up); the top TOP under the practice rule; gseapy prerank
on the full score ranking. A cluster's reference type is the majority of its cells' mapped CellTypist labels. Truth:
Monaco et al. 2019's sorted-cell bulk (data_sources.yaml), each type the mean of the populations BULK maps to it: the
type's genes with at least MIN_TPM and BULK_FOLD times the rest's profile weighted by its cells' types, and prerank on
log2((type + 1) / (rest + 1)). The curated markers of config.yaml are not used: they are the Seurat PBMC3k tutorial's,
read from Wilcoxon output on PBMC3k. Readouts per dataset, method and resolution, and the stability of a cell's top
TOP and NES between consecutive resolutions.

    python -m fetch_data --arm clustering              # from reproducibility/; the sorted-cell bulk, 15 MB
    python -m clustering.multires_gaps                 # about 15 minutes
    python -m clustering.multires_gaps --plot-only     # redraw from the CSVs
"""

import argparse
import tarfile

import anndata as ad
import gseapy as gp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr

import plot_style
from clustering import concordance as cc
from clustering import multires as mr
from clustering.celltype_labels import RAW, dataset_paths
from clustering.heldout_fp import de_output
from clustering.replication import save
from evaluation_utils import _get_rank_genes_df
from method_colors import MEDIANPROPS, method_color, method_draw_order, method_label
from paths import arm_dir, fig_name, require_input, results_dir

METHODS = mr.METHODS
TOP = mr.TOP
MIN_DETECT = 5          # cells detecting a gene on each side
EQUAL_MEAN = 0.1        # |log2 mean ratio| within which a gene's means agree (7%)
MIN_TPM, BULK_FOLD = 1.0, 1.5
MONACO = arm_dir(__file__).parent / 'data' / 'GSE107011_Processed_data_TPM.txt.gz'
BULK = {'Naive CD4 T': ['CD4_naive'], 'Memory CD4 T': ['Th1', 'Th1/Th17', 'Th17', 'Th2', 'TFH', 'Treg'],
        'CD8 T': ['CD8_naive', 'CD8_CM', 'CD8_EM', 'CD8_TE'], 'NK': ['NK'], 'B': ['B_naive', 'B_NSM', 'B_SM', 'B_Ex'],
        'CD14 Mono': ['C_mono'], 'FCGR3A Mono': ['NC_mono'], 'DC': ['mDC']}
GAPS = {'gap_equal_mean': 'Variance gap at equal mean\n(median |log2 variance ratio|)',
        'gap_all': 'Variance gap, all genes\n(median |log2 variance ratio|)',
        'gap_depth': 'Depth gap\n(|log2 median counts ratio|)'}
RATIO = {'gap_equal_mean': 'Variance ratio', 'gap_all': 'Variance ratio, all genes'}   # the first at equal mean
# shown as 2 ** the gap, the median ratio of the larger variance to the smaller (from 1, as Fig 1's Var(Y) / Var(X);
# with an even number of genes, the geometric mean of the two middle ratios)
THREADS = 4             # gseapy's threads per prerank run; its results do not depend on them
OUT = results_dir(__file__) / 'multires_gaps'


def bulk_profiles():
    """Each curated type's sorted-cell bulk TPM by Ensembl ID: the mean over donors of each population, then over the
    populations BULK maps to the type."""
    tpm = pd.read_csv(require_input(MONACO, what='the Monaco et al. 2019 sorted-cell bulk',
                                    source='python -m fetch_data --arm clustering'), sep='\t', index_col=0)
    tpm.index = tpm.index.str.split('.').str[0]
    tpm = tpm.groupby(level=0).sum()
    population = tpm.columns.str.split('_', n=1).str[1]
    means = tpm.T.groupby(population.to_numpy()).mean().T
    return pd.DataFrame({t: means[p].mean(axis=1) for t, p in BULK.items()})


def ensembl_ids(name, genes):
    """Each of the arm's gene names' Ensembl ID, from the dataset's 10x annotation; NaN where none."""
    if name == 'PBMC10k':
        with h5py.File(require_input(RAW, what="the 5' PBMC 10k matrix", source='python -m fetch_data --arm citeseq'),
                       'r') as f:
            ids = f['matrix/features/id'][:].astype(str)
            names = f['matrix/features/name'][:].astype(str)
            kinds = f['matrix/features/feature_type'][:].astype(str)
        table = pd.Series(ids, index=ad.utils.make_index_unique(pd.Index(names)))[kinds == 'Gene Expression']
    else:
        config_path, _ = dataset_paths()[name]
        tar = require_input(yaml.safe_load(open(config_path))['adata_path'], what='the PBMC3k 10x archive',
                            source='python -m fetch_data --arm clustering')
        with tarfile.open(tar) as tf:
            g = pd.read_csv(tf.extractfile(next(m for m in tf.getmembers() if m.name.endswith('genes.tsv'))),
                            sep='\t', header=None)
        table = pd.Series(g[0].astype(str).to_numpy(), index=ad.utils.make_index_unique(pd.Index(g[1].astype(str))))
    return table.reindex(genes)


def moments(cp, lab, ids):
    """Per cluster and its rest: cells, and each gene's mean, variance and number of cells detecting it."""
    g = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), pd.Index(ids).get_indexer(lab))),
                      shape=(len(lab), len(ids)))
    n = np.asarray(g.sum(0)).ravel()
    s1 = np.asarray((g.T @ cp).todense())
    s2 = np.asarray((g.T @ cp.multiply(cp)).todense())
    det = np.asarray((g.T @ (cp > 0).astype(np.float64)).todense())
    out = {}
    for i, c in enumerate(ids):
        nr = len(lab) - n[i]
        m1, mr_ = s1[i] / n[i], (s1.sum(0) - s1[i]) / nr
        v1 = (s2[i] - n[i] * m1 ** 2) / max(n[i] - 1, 1)
        vr = (s2.sum(0) - s2[i] - nr * mr_ ** 2) / max(nr - 1, 1)
        out[c] = (m1, v1, det[i], mr_, vr, det.sum(0) - det[i])
    return out


def gaps(m1, v1, d1, mr_, vr, dr):
    """The two variance gaps of a comparison, with their numbers of genes and the twofold share at equal mean."""
    keep = (d1 >= MIN_DETECT) & (dr >= MIN_DETECT) & (v1 > 0) & (vr > 0)
    lv = np.abs(np.log2(v1[keep] / vr[keep]))
    equal = np.abs(np.log2(m1[keep] / mr_[keep])) < EQUAL_MEAN
    return {'gap_equal_mean': np.median(lv[equal]) if equal.any() else np.nan, 'genes_equal_mean': int(equal.sum()),
            'twofold_equal_mean': np.mean(lv[equal] >= 1) if equal.any() else np.nan,
            'gap_all': np.median(lv) if keep.any() else np.nan, 'genes_all': int(keep.sum())}


def prerank(key, names, values):
    """concordance.prerank's run on one ranking (its settings, THREADS threads), keyed by its comparison and source."""
    r = pd.DataFrame({'names': names, 'value': values})
    r = r[np.isfinite(r['value'])].sort_values(['value', 'names'], ascending=[False, True])
    res = gp.prerank(rnk=r.reset_index(drop=True), gene_sets=str(cc.GMT), permutation_num=cc.PERMUTATIONS,
                     seed=cc.SEED, min_size=cc.MIN_SIZE, max_size=cc.MAX_SIZE, threads=THREADS, outdir=None,
                     no_plot=True, verbose=False).res2d
    return pd.DataFrame({'set': res['Term'].to_numpy(), 'nes': res['NES'].astype(float).to_numpy(),
                         'fdr': res['FDR q-val'].astype(float).to_numpy()}).assign(**key)


def run():
    """Per comparison its gaps; per comparison and method its calls, top TOP and agreement with the truths; the GSEA
    tasks."""
    profiles = bulk_profiles()
    comps, per, tops, tasks, bulk_true = [], [], {}, [], {}
    for name in dataset_paths():
        adata, cp, coarse, keys, _ = mr.load(name)
        ens = ensembl_ids(name, adata.var_names)
        matched = ens.notna() & ens.isin(profiles.index)
        print(f'  {name}: {matched.sum()} of {len(ens)} genes matched to the sorted-cell bulk', flush=True)
        bulk = profiles.reindex(ens[matched].to_numpy())
        bulk.index = adata.var_names[matched.to_numpy()]
        types = coarse.fillna('(unmapped)').to_numpy()
        depth = adata.obs['n_counts'].to_numpy(dtype=float)
        for key in keys:
            lab = adata.obs[key].astype(str).to_numpy()
            ids = sorted(np.unique(lab), key=lambda v: int(v) if v.isdigit() else v)
            mom = moments(cp, lab, ids)
            for c in ids:
                own = lab == c
                counts = pd.Series(types[own]).value_counts().rename_axis('type').reset_index()
                top = counts.sort_values(['count', 'type'], ascending=[False, True])['type'].iloc[0]
                reference = None if top == '(unmapped)' else top
                ratio = np.median(depth[own]) / np.median(depth[~own])
                comps.append({'dataset': name, 'resolution': key, 'clusters': len(ids), 'cluster': c,
                              'cells': int(own.sum()), 'reference': reference, 'depth_ratio': ratio,
                              'gap_depth': abs(np.log2(ratio)), **gaps(*mom[c])})
                comp = {'dataset': name, 'resolution': key, 'cluster': c}
                judged = reference in BULK
                if judged:
                    rest = pd.Series(types[~own]).value_counts()
                    rest = rest[rest.index.isin(list(BULK))]
                    rest_profile = (bulk[rest.index] * rest.to_numpy()).sum(axis=1) / rest.sum()
                    t = bulk[reference]
                    bulk_true[name, key, c] = set(bulk.index[(t >= MIN_TPM) & (t >= BULK_FOLD * rest_profile)])
                    tasks.append(({**comp, 'source': 'bulk'}, bulk.index.to_numpy(),
                                  np.log2((t.to_numpy() + 1) / (rest_profile.to_numpy() + 1))))
                for m in METHODS:
                    res = _get_rank_genes_df(adata, c, f'{m}_{key}')
                    sig = res['pvals_adj'] < 0.05
                    full = mr.ranked(adata, c, f'{m}_{key}')
                    tops[name, key, c, m] = full[:TOP]
                    row = {**comp, 'method': m, 'calls': int(sig.sum()), 'calls_up': int((sig & (res['scores'] > 0)).sum()),
                           'top': ';'.join(full[:TOP])}
                    judgeable = [g for g in full[:TOP] if g in bulk.index]
                    row['bulk_judged'] = len(judgeable)
                    row['bulk_precision'] = (np.mean([g in bulk_true[name, key, c] for g in judgeable])
                                             if judged and judgeable else np.nan)
                    per.append(row)
                    r = res[['names', 'scores']].astype({'scores': float})
                    tasks.append(({**comp, 'source': m}, r['names'].astype(str).to_numpy(), r['scores'].to_numpy()))
        print(f'  {name}: {sum(1 for c in comps if c["dataset"] == name)} comparisons', flush=True)
    print(f'  {len(tasks)} prerank runs', flush=True)
    gsea = pd.concat([prerank(*task) for task in tasks], ignore_index=True)
    gsea = gsea[['dataset', 'resolution', 'cluster', 'source', 'set', 'nes', 'fdr']]
    gsea = gsea.sort_values(['dataset', 'resolution', 'cluster', 'source', 'set'], ignore_index=True)
    per = pd.DataFrame(per)
    per['nes_bulk_spearman'] = [nes_agreement(gsea, r.dataset, r.resolution, r.cluster, r.method)
                                for r in per.itertuples()]
    return pd.DataFrame(comps), per, gsea


def nes_agreement(gsea, dataset, resolution, cluster, method):
    """Spearman correlation of a comparison's NES under the method and under the sorted-cell bulk, over the sets both
    report; NaN where the cluster is not judged."""
    g = gsea[(gsea['dataset'] == dataset) & (gsea['resolution'] == resolution) & (gsea['cluster'] == cluster)]
    w = g.pivot(index='set', columns='source', values='nes')
    if 'bulk' not in w:
        return np.nan
    w = w[[method, 'bulk']].dropna()
    return spearmanr(w[method], w['bulk']).statistic if len(w) > 2 else np.nan


def stability(comps, per, gsea):
    """Per dataset, method and pair of consecutive resolutions, averaged over cells: the Jaccard index of the top TOP
    of a cell's two clusters, and the Spearman correlation of their NES."""
    rows = []
    for name in dataset_paths():
        config = yaml.safe_load(open(dataset_paths()[name][0]))
        keys = [k for _, k in config['leiden_resolutions']]
        obs = labels_of(name, keys)
        n = comps[comps['dataset'] == name].groupby('resolution')['clusters'].first()
        for a, b in zip(keys[:-1], keys[1:]):
            pairs = obs.groupby([a, b]).size()
            for m in METHODS:
                top = per[(per['dataset'] == name) & (per['method'] == m)].set_index(['resolution', 'cluster'])['top']
                nes = (gsea[(gsea['dataset'] == name) & (gsea['source'] == m)]
                       .pivot_table(index='set', columns=['resolution', 'cluster'], values='nes'))
                jac, rho = [], []
                for (ca, cb), cells in pairs.items():
                    ta, tb = set(str(top[a, ca]).split(';')) - {''}, set(str(top[b, cb]).split(';')) - {''}
                    jac.append((len(ta & tb) / len(ta | tb) if ta | tb else np.nan, cells))
                    w = nes[[(a, ca), (b, cb)]].dropna()
                    rho.append((spearmanr(w.iloc[:, 0], w.iloc[:, 1]).statistic if len(w) > 2 else np.nan, cells))
                rows.append({'dataset': name, 'method': m, 'from': a, 'to': b, 'clusters_from': int(n[a]),
                             'clusters_to': int(n[b]), 'top_jaccard': weighted(jac), 'nes_spearman': weighted(rho)})
    return pd.DataFrame(rows)


def labels_of(name, keys):
    """Every cell's cluster at each resolution, as strings."""
    config_path, out = dataset_paths()[name]
    with h5py.File(de_output(yaml.safe_load(open(config_path))['adata_path'], str(out)), 'r') as f:
        obs = ad.io.read_elem(f['obs'])
    return obs[keys].astype(str)


def weighted(values):
    """The cell-weighted mean of (value, cells) pairs, NaN values left out."""
    v = np.array([x for x, _ in values], dtype=float)
    w = np.array([c for _, c in values], dtype=float)
    ok = np.isfinite(v)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def summarise(comps, per):
    """Per dataset, resolution and method: median calls, mean bulk precision and NES agreement; and
    per dataset and method, the Spearman correlation over comparisons between calls and each gap."""
    d = per.merge(comps, on=['dataset', 'resolution', 'cluster'])
    g = d.groupby(['dataset', 'resolution', 'clusters', 'method'])
    summary = pd.DataFrame({'comparisons': g.size(), 'median_calls': g['calls'].median(),
                            'median_calls_up': g['calls_up'].median(), 'bulk_precision': g['bulk_precision'].mean(),
                            'bulk_clusters': g['bulk_precision'].count(),
                            'nes_bulk_spearman': g['nes_bulk_spearman'].mean()}).reset_index()
    gs = comps.groupby(['dataset', 'resolution', 'clusters'])
    gap_summary = gs[list(GAPS) + ['twofold_equal_mean']].median().reset_index()
    rows = []
    for (name, m), e in d.groupby(['dataset', 'method']):
        for gap in GAPS:
            for calls in ('calls', 'calls_up'):
                ok = e[gap].notna()
                rows.append({'dataset': name, 'method': m, 'gap': gap, 'calls': calls, 'comparisons': int(ok.sum()),
                             'spearman': spearmanr(e.loc[ok, gap], e.loc[ok, calls]).statistic})
    return summary.sort_values(['dataset', 'clusters', 'method'], ignore_index=True), \
        gap_summary.sort_values(['dataset', 'clusters'], ignore_index=True), pd.DataFrame(rows)


def positions(table, name):
    """The dataset's resolutions in order of their number of clusters, with x positions."""
    n = table[table['dataset'] == name].groupby('resolution')['clusters'].first().sort_values()
    return n, {r: i for i, r in enumerate(n.index)}


def ratio_ticks(ax):
    """A variance-ratio axis from 1, ticked at 1 and every whole number (every other one past 6)."""
    top = ax.get_ylim()[1]
    ax.set_yticks(np.arange(1, top + 1e-9, 1 if top <= 6 else 2))


def shown(values, gap):
    """A variance gap as its ratio (RATIO), any other gap as it is."""
    return 2 ** values if gap in RATIO else values


def draw_gaps(ax, comps, name, gap):
    """One gap's values as a box at each resolution, every cluster a point over its box; a variance gap as its ratio."""
    n, x = positions(comps, name)
    d = comps[comps['dataset'] == name]
    rng = np.random.default_rng(0)
    for r, i in x.items():
        v = shown(d.loc[d['resolution'] == r, gap].dropna().to_numpy(), gap)
        bp = ax.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False, medianprops=MEDIANPROPS,
                        manage_ticks=False)
        for box in bp['boxes']:
            box.set_facecolor('0.8')
            box.set_edgecolor('none')
        plot_style.strip(ax, i, v, '0.55', 0.5, rng)
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.to_numpy())
    ax.set_xlim(-0.5, len(n) - 0.5)   # one equal column per resolution, so a row of plots can sit above them
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(RATIO.get(gap, GAPS[gap]))
    ax.set_ylim(bottom=1 if gap in RATIO else 0)
    if gap in RATIO:
        ratio_ticks(ax)
    plot_style.boxplot_grid(ax)


DATASET_GREY = {'PBMC10k': '0.2', 'PBMC3k': '0.65'}
COARSE_COLOURS = ('#9467bd', '#e377c2', '#7f7f7f')   # purple, pink, grey: away from the method colours and Fig 2a's


def umap_clusters(name):
    """The dataset's UMAP and its Leiden clusters at each resolution, coarse to fine, from clustering/de.py's output."""
    config_path, out = dataset_paths()[name]
    config = yaml.safe_load(open(config_path))
    keys = [k for _, k in config['leiden_resolutions']]
    with h5py.File(require_input(de_output(config['adata_path'], str(out)), what=f'the {name} DE output',
                                 source='python -m clustering.de'), 'r') as f:
        obs = ad.io.read_elem(f['obs'])
        umap = np.asarray(ad.io.read_elem(f['obsm/X_umap']))
    return umap, obs[sorted(keys, key=lambda k: obs[k].nunique())]


def draw_clusters(ax, umap, labels):
    """The cells on the UMAP coloured by their cluster: a few clusters in COARSE_COLOURS, more in scanpy's default palette."""
    from scanpy.plotting.palettes import default_20, default_102
    codes = pd.Categorical(labels).codes
    n = codes.max() + 1
    palette = COARSE_COLOURS if n <= len(COARSE_COLOURS) else default_20 if n <= 20 else default_102
    ax.scatter(umap[:, 0], umap[:, 1], s=0.2, lw=0, c=[palette[c] for c in codes], rasterized=True)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
        sp_.set_color('0.6')
        sp_.set_linewidth(0.6)


def draw_gap_vs_depth(ax, comps, gap='gap_equal_mean', names=None):
    """Each comparison's variance ratio against its |log2 depth ratio|, every dataset or those in names, each with its
    Spearman correlation."""
    if names is not None:
        comps = comps[comps['dataset'].isin(names)]
    for name, d in comps.groupby('dataset', sort=False):
        rho = spearmanr(d['gap_depth'], d[gap]).statistic
        ax.scatter(d['gap_depth'], shown(d[gap], gap), s=6, lw=0, alpha=0.8, color=DATASET_GREY[name],
                   label=f'{name}, $\\rho$ = {rho:.2f}')
    ax.set_xlabel('|log2 depth ratio|')
    ax.set_ylabel(RATIO[gap])
    ax.set_xlim(left=0)
    top = shown(comps[gap].max(), gap)
    ax.set_ylim(1, 1 + 1.3 * (top - 1))   # room above the points for a legend
    ratio_ticks(ax)


def draw_calls_vs_gap(ax, comps, per, name, gap, calls='calls'):
    """Each comparison's number of calls against its gap, one colour per method, log axis."""
    d = per[per['dataset'] == name].merge(comps, on=['dataset', 'resolution', 'cluster'])
    for m in method_draw_order(METHODS):
        e = d[d['method'] == m]
        ax.scatter(e[gap], e[calls] + 1, s=5, lw=0, alpha=0.8, color=method_color(m), label=method_label(m))
    ax.set_yscale('log')
    ax.set_xlabel(GAPS[gap].split('\n')[0])
    ax.set_ylabel('Significant genes + 1' if calls == 'calls' else 'Upregulated genes + 1')


def draw_by_resolution(ax, summary, name, column, ylabel, ylim=None, log=False):
    """A per-resolution readout against the number of clusters, one line per method."""
    d = summary[summary['dataset'] == name]
    for m in method_draw_order(METHODS):
        e = d[d['method'] == m].sort_values('clusters')
        ax.plot(range(len(e)), e[column], marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                lw=plot_style.ERRORBAR['lw'], color=method_color(m), label=method_label(m))
    n = d.groupby('resolution')['clusters'].first().sort_values()
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.to_numpy())
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(*ylim)


def draw_stability(ax, stab, name, column, ylabel):
    """A stability between consecutive resolutions, one line per method."""
    d = stab[stab['dataset'] == name]
    for m in method_draw_order(METHODS):
        e = d[d['method'] == m].sort_values('clusters_from')
        ax.plot(range(len(e)), e[column], marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                lw=plot_style.ERRORBAR['lw'], color=method_color(m), label=method_label(m))
    e = d[d['method'] == METHODS[0]].sort_values('clusters_from')
    ax.set_xticks(range(len(e)))
    ax.set_xticklabels([f'{a}→{b}' for a, b in zip(e['clusters_from'], e['clusters_to'])])
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)


READOUTS = {'calls_by_resolution': ('median_calls', 'Median significant genes', None, True),
            'top_bulk': ('bulk_precision', f'Top {TOP} confirmed by sorted bulk', (0, 1), False),
            'gsea_bulk': ('nes_bulk_spearman', 'NES correlation with sorted bulk', (-0.2, 1), False)}
STABILITY = {'top_stability': ('top_jaccard', f'Top {TOP} Jaccard,\nconsecutive resolutions'),
             'gsea_stability': ('nes_spearman', 'NES correlation,\nconsecutive resolutions')}


def plot(comps, per, summary, stab):
    """One plot per readout and dataset."""
    plot_style.use()
    for name in dataset_paths():
        tag = name.lower()
        for gap in GAPS:
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_gaps(ax, comps, name, gap)
            ax.set_title(name)
            save(fig, OUT / fig_name(f'{tag}_{gap}'), key=False)
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_calls_vs_gap(ax, comps, per, name, gap)
            ax.set_title(name)
            save(fig, OUT / fig_name(f'{tag}_calls_vs_{gap}'))
        for stem, (column, ylabel, ylim, log) in READOUTS.items():
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_by_resolution(ax, summary, name, column, ylabel, ylim, log)
            ax.set_title(name)
            save(fig, OUT / fig_name(f'{tag}_{stem}'))
        for stem, (column, ylabel) in STABILITY.items():
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_stability(ax, stab, name, column, ylabel)
            ax.set_title(name)
            save(fig, OUT / fig_name(f'{tag}_{stem}'))
    for gap in ('gap_equal_mean', 'gap_all'):
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_gap_vs_depth(ax, comps, gap)
        ax.legend(fontsize=plot_style.SMALL, frameon=False, loc='upper left')
        save(fig, OUT / fig_name(f'{gap}_vs_depth'), key=False)


def main():
    parser = argparse.ArgumentParser(description='Variance gaps, calls, top markers and GSEA across Leiden resolutions')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ('comparisons', 'per_method', 'gsea', 'summary', 'gap_summary', 'correlations', 'stability')
    csv = {k: OUT / f'{k}.csv' for k in names}
    if args.plot_only:
        t = {k: pd.read_csv(require_input(p, what=f'the multires_gaps {k}', source='this script without --plot-only'),
                            dtype={'cluster': str})
             for k, p in csv.items()}
    else:
        comps, per, gsea = run()
        summary, gap_summary, corr = summarise(comps, per)
        t = {'comparisons': comps, 'per_method': per, 'gsea': gsea, 'summary': summary, 'gap_summary': gap_summary,
             'correlations': corr, 'stability': stability(comps, per, gsea)}
        for k, v in t.items():
            v.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(t['comparisons'], t['per_method'], t['summary'], t['stability'])


if __name__ == '__main__':
    main()
