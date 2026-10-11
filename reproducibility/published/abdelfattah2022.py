#!/usr/bin/env python3
"""Abdelfattah et al. 2022 (Nat Commun 13:767), glioma myeloid cells: their Fig. 4d pathways under the authors'
procedure and under LN, with each cluster's depth.

The authors subclustered 83,479 myeloid cells into MC01-MC09, ranked every gene of each cluster against the other myeloid
cells by presto's Wilcoxon AUC and ran fgsea on msigdbr's Hallmark sets (their R/Figure4.R). Here: their cells and labels
(data_sources.yaml), the genes both of the study's 10x references share, detected in at least MIN_CELLS cells; each
cluster against the rest ranked by the AUC (as presto computes it, on log1p CP10K, uncentred as they ranked it), by LN's
score and by the log1p t-test's score; their fgsea call on each ranking (published/R/abdelfattah2022_fgsea.R).
Readouts per cluster and method: agreement with the signs of the published heatmap (read from the
figure); against the cluster's depth ratio, the share of its expressed genes ranked up (detected in at least EXPRESSED
of the cluster's or the rest's cells) and the share of the 50 sets with positive NES; the Fig. 4d cells where the
authors' procedure and LN differ in sign with at least one called; and, for MC03, detection and mean level of the
TNF-alpha via NF-kB genes. The myeloid cells' UMAP, by cluster and by depth, is drawn from
published/R/abdelfattah2022_umap.R's output (Seurat and harmony, as the paper describes its embedding; the authors' own
myeloid embedding is not public).

    python -m fetch_data --arm published               # from reproducibility/, 2.5 GB
    python -m published.abdelfattah2022 --rankings     # 1, the rankings and per-cluster tables; a few minutes
    Rscript published/R/abdelfattah2022_fgsea.R        # 2, fgsea on every ranking; a minute
    Rscript published/R/abdelfattah2022_umap.R         # 3, the UMAP; several minutes
    python -m published.abdelfattah2022                # 4, the readouts and plots; seconds
    python -m published.abdelfattah2022 --variance     # each cluster's variance ratio at equal mean (variance.csv), as
                                                       # Fig 2e's (clustering.multires_gaps); a minute or two
    python -m published.abdelfattah2022 --claims       # the paper's Fig. 4d claims under the authors' procedure, the
                                                       # Wilcoxon z and LN (claims.csv); seconds
    python -m published.abdelfattah2022 --calls        # each test's Bonferroni calls per cluster and the share up
                                                       # (calls.csv) and their plot; a few minutes (--plot-only: redraw)
"""

import argparse
import glob

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from matplotlib import cbook
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress, norm, rankdata, spearmanr

import plot_style
from clustering import multires_gaps as mg
from clustering.concordance import GMT
from evaluation_utils import _get_rank_genes_df
from lntest import rank_genes_groups_ln
from method_colors import method_color, method_label
from paths import arm_dir, fig_name, require_input, results_dir

DATA = arm_dir(__file__).parent / 'data'
COUNTS = DATA / 'GSE182109'
META = DATA / 'Meta_Data_GBMatlas.txt'
OUT = results_dir(__file__)
MIN_CELLS = 3
EXPRESSED = 0.05    # share of the cluster's or the rest's cells detecting a gene
Q = 0.05
METHODS = ('wilcoxon', 'ln', 't_test')
# Each figure sets one comparator against LN: the Wilcoxon AUC (the authors' ranking), the Wilcoxon's tie-corrected z
# (scanpy's), or the log1p t-test's score; at the gene level each method's per-gene value (GENE).
COMPARATORS = ('wilcoxon', 'wilcoxon_z', 't_test')
NAME = {'wilcoxon': 'Wilcoxon', 'wilcoxon_z': 'Wilcoxon z', 'ln': method_label('ln'), 't_test': method_label('t_test')}
TITLE = {**NAME, 'wilcoxon': 'Wilcoxon AUC', 'wilcoxon_z': 'Wilcoxon'}   # panel titles: the z is scanpy's Wilcoxon score
GENE = {'wilcoxon': ('wilcoxon', 'AUC', 0.5), 'wilcoxon_z': ('wilcoxon_z', 'Score', 0.0),
        't_test': ('t_lfc', 'LFC', 0.0), 'ln': ('ln', 'Score', 0.0)}
COLOUR = {'wilcoxon': 'wilcoxon', 'wilcoxon_z': 'wilcoxon', 't_test': 't_test', 't_lfc': 't_test', 'ln': 'ln',
          'ln_lfc': 'ln'}     # each value's method, for its colour
FOCUS, FOCUS_SET = 'MC03', 'TNF-alpha Signaling via NF-kB'   # the focus set's genes from GMT (Enrichr's copy)
# msigdbr's names of the sets shown, as the figures name them
DISPLAY = {'HALLMARK_ADIPOGENESIS': 'Adipogenesis', 'HALLMARK_INTERFERON_ALPHA_RESPONSE': 'Interferon Alpha Response',
           'HALLMARK_MITOTIC_SPINDLE': 'Mitotic Spindle', 'HALLMARK_MTORC1_SIGNALING': 'mTORC1 Signaling',
           'HALLMARK_MYC_TARGETS_V1': 'Myc Targets V1', 'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Oxidative Phosphorylation',
           'HALLMARK_INTERFERON_GAMMA_RESPONSE': 'Interferon Gamma Response', 'HALLMARK_G2M_CHECKPOINT': 'G2-M Checkpoint',
           'HALLMARK_E2F_TARGETS': 'E2F Targets', 'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNF-alpha Signaling via NF-kB',
           'HALLMARK_P53_PATHWAY': 'p53 Pathway', 'HALLMARK_ALLOGRAFT_REJECTION': 'Allograft Rejection',
           'HALLMARK_KRAS_SIGNALING_DN': 'KRAS Signaling Dn', 'HALLMARK_PANCREAS_BETA_CELLS': 'Pancreas Beta Cells',
           'HALLMARK_INFLAMMATORY_RESPONSE': 'Inflammatory Response', 'HALLMARK_HYPOXIA': 'Hypoxia',
           'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'Epithelial Mesenchymal Transition'}
# The paper's names for the clusters (MC1 (i-Mic) ... MC9 (p-Mac)); the deposited metadata calls MC09 'Proliferating'.
PAPER = {'MC01': 'MC1 i-Mic', 'MC02': 'MC2 h-Mic', 'MC03': 'MC3 s-Mac1', 'MC04': 'MC4 MDSC', 'MC05': 'MC5 s-Mac2',
         'MC06': 'MC6 AP-Mic', 'MC07': 'MC7 a-Mic', 'MC08': 'MC8 DC', 'MC09': 'MC9 p-Mac'}
LABEL = {c: name.split()[0] for c, name in PAPER.items()}    # MC1 ... MC9, as the plots label them
# The interpretation the rerun contradicts: MC3 (s-Mac1) as IFN-responsive, and MDSCs (MC4) lower than it in MYC targets,
# IFN-gamma response and proliferation (the paper's text on its Fig. 4d).
CONTRAST = ('MC03', 'MC04')
CONTRAST_SETS = {'Interferon Alpha Response': 'IFN-α', 'Interferon Gamma Response': 'IFN-γ', 'Myc Targets V1': 'MYC',
                 'G2-M Checkpoint': 'G2-M', 'E2F Targets': 'E2F'}
SHARED_VMAX = 5     # the heatmaps' shared colour range, the published figure's; the Wilcoxon's larger NES saturate

# The published Fig. 4d heatmap's signs, read from the figure; KRAS Signaling Dn and Pancreas Beta Cells sit near zero
# in every cluster and are not read.
PUBLISHED = {
    'Adipogenesis': '--+-++--+', 'Interferon Alpha Response': '--+--+-++', 'Mitotic Spindle': '--+--+--+',
    'mTORC1 Signaling': '--++++-++', 'Myc Targets V1': '--+-++-++', 'Oxidative Phosphorylation': '--+-++-++',
    'Interferon Gamma Response': '--+-++-++', 'G2-M Checkpoint': '--+-++-++', 'E2F Targets': '--+-++-++',
    'TNF-alpha Signaling via NF-kB': '+-+++--+-', 'p53 Pathway': '+-++++-++', 'Allograft Rejection': '+-+-++-+-',
    'Inflammatory Response': '+-+++----', 'Hypoxia': '--+++----', 'Epithelial Mesenchymal Transition': '--+++----'}
ROWS = list(PUBLISHED)[:12] + ['KRAS Signaling Dn', 'Pancreas Beta Cells'] + list(PUBLISHED)[12:]
SHORT = {'Interferon Alpha Response': 'IFN-α response', 'Mitotic Spindle': 'Mitotic spindle',
         'mTORC1 Signaling': 'mTORC1 signaling', 'Myc Targets V1': 'MYC targets V1', 'Oxidative Phosphorylation': 'OXPHOS',
         'Interferon Gamma Response': 'IFN-γ response', 'G2-M Checkpoint': 'G2-M checkpoint', 'E2F Targets': 'E2F targets',
         'TNF-alpha Signaling via NF-kB': 'TNF-α via NF-κB', 'p53 Pathway': 'p53 pathway',
         'Allograft Rejection': 'Allograft rejection', 'KRAS Signaling Dn': 'KRAS signaling down',
         'Pancreas Beta Cells': 'Pancreas β cells', 'Inflammatory Response': 'Inflammatory response',
         'Epithelial Mesenchymal Transition': 'EMT'}   # the heatmaps' row labels
CLUSTERS = [f'MC0{i}' for i in range(1, 10)]

def load():
    """The authors' labelled myeloid cells: raw counts over the genes both references share, with their metadata."""
    meta = pd.read_csv(require_input(META, what='the authors\' per-cell metadata', source='python -m fetch_data --arm published'),
                       sep='\t', index_col=0, low_memory=False)
    meta = meta[meta['SubCluster'].astype(str).isin(CLUSTERS)]
    require_input(COUNTS, what='the GSE182109 matrices', source='python -m fetch_data --arm published')
    parts, cells, missing = [], [], 0
    for gsm, m in meta.groupby('GSMID'):
        x = scipy.io.mmread(glob.glob(str(COUNTS / f'{gsm}_*matrix.mtx.gz'))[0]).T.tocsr()
        barcodes = pd.read_csv(glob.glob(str(COUNTS / f'{gsm}_*barcodes.tsv.gz'))[0], header=None)[0].astype(str)
        genes = pd.read_csv(glob.glob(str(COUNTS / f'{gsm}_*features.tsv.gz'))[0], header=None, sep='\t')[1].astype(str)
        first = ~genes.duplicated().to_numpy()
        idx = pd.Index(barcodes).get_indexer(m['barcode'].astype(str))
        missing += int((idx < 0).sum())
        parts.append((pd.Index(genes[first]), x[idx[idx >= 0]][:, np.flatnonzero(first)]))
        cells.append(m.index[idx >= 0])
    common = parts[0][0]
    for g, _ in parts[1:]:
        common = common.intersection(g)
    x = sp.vstack([x[:, g.get_indexer(common)] for g, x in parts]).tocsr().astype(np.float64)
    obs = meta.loc[np.concatenate(cells), ['GSMID', 'nCount_RNA', 'SubCluster', 'SubAssignment']]
    keep = np.asarray((x > 0).sum(0)).ravel() >= MIN_CELLS
    print(f'  {x.shape[0]} cells ({missing} labelled cells absent from the matrices), {len(common)} shared genes, '
          f'{keep.sum()} detected in at least {MIN_CELLS} cells')
    return x[:, keep], common[keep], obs


def auc(cp, labels, k):
    """presto's AUC of each of k groups against the rest, per gene, from a cells-by-genes CSC matrix; zeros tie."""
    n = len(labels)
    nk = np.bincount(labels, minlength=k).astype(float)
    out = np.empty((k, cp.shape[1]))
    for j in range(cp.shape[1]):
        s, e = cp.indptr[j], cp.indptr[j + 1]
        lab = labels[cp.indices[s:e]]
        n0 = n - (e - s)
        r = rankdata(cp.data[s:e]) + n0
        ranksum = np.bincount(lab, weights=r, minlength=k) + (nk - np.bincount(lab, minlength=k)) * (n0 + 1) / 2
        out[:, j] = (ranksum - nk * (nk + 1) / 2) / (nk * (n - nk))
    return out


def normalised(x, groups, genes):
    """The cells as an AnnData of CP10K and log1p CP10K, grouped by cluster, and the clusters in order."""
    cp = sp.csr_matrix(sp.diags(1e4 / np.asarray(x.sum(1)).ravel()) @ x, dtype=np.float32)
    lg = cp.copy()
    lg.data = np.log1p(lg.data)
    cats = sorted(set(groups))
    a = ad.AnnData(X=cp, obs=pd.DataFrame({'g': pd.Categorical(groups, categories=cats)},
                                          index=np.arange(len(groups)).astype(str)), var=pd.DataFrame(index=genes))
    a.layers['norm_counts'], a.layers['log1p_norm'] = cp, lg
    return a, cats


def rankings(x, groups, genes):
    """Per group, against the rest, over genes: the AUC, the Wilcoxon's tie-corrected z, LN's score and LFC, and the
    log1p t-test's score and LFC (scanpy's)."""
    a, cats = normalised(x, groups, genes)
    labels = pd.Categorical(groups, categories=cats).codes.astype(int)
    a_ = auc(a.layers['norm_counts'].tocsc(), labels, len(cats))
    rank_genes_groups_ln(a, 'g', sparse=True, key_added='ln', layer='norm_counts')
    sc.tl.rank_genes_groups(a, 'g', method='t-test', layer='log1p_norm', use_raw=False, key_added='t_test')
    sc.tl.rank_genes_groups(a, 'g', method='wilcoxon', layer='log1p_norm', use_raw=False, tie_correct=True,
                            key_added='wilcoxon_z')
    col = lambda c, key, f: _get_rank_genes_df(a, c, key).set_index('names').reindex(genes)[f].astype(float)
    return {c: {'wilcoxon': pd.Series(a_[i], index=genes), 'wilcoxon_z': col(c, 'wilcoxon_z', 'scores'),
                'ln': col(c, 'ln', 'scores'), 'ln_lfc': col(c, 'ln', 'logfoldchanges'),
                't_test': col(c, 't_test', 'scores'), 't_lfc': col(c, 't_test', 'logfoldchanges')}
            for i, c in enumerate(cats)}


def run():
    """Every cluster's ranking per method (and LN's LFC), each gene flagged expressed (detected in at least
    EXPRESSED of the cluster's or the rest's cells); per cluster and method the share of expressed genes ranked up (AUC
    above 0.5, a positive score); the clusters' depths; the focus cluster's genes of the focus set."""
    x, genes, obs = load()
    lab = obs['SubCluster'].astype(str).to_numpy()
    depth = np.asarray(x.sum(1)).ravel()
    detected = (x > 0).astype(np.float64).tocsr()
    rows, up, clusters = [], [], []
    for c, rk in rankings(x, lab, genes).items():
        own = lab == c
        expressed = ((np.asarray(detected[own].mean(0)).ravel() >= EXPRESSED) |
                     (np.asarray(detected[~own].mean(0)).ravel() >= EXPRESSED))
        rows += [pd.DataFrame({'cluster': c, 'method': m, 'gene': genes, 'value': r.to_numpy(), 'expressed': expressed})
                 for m, r in rk.items()]
        for m, r in rk.items():
            if m.endswith('_lfc'):
                continue
            v = r.to_numpy()[expressed]
            v = v[np.isfinite(v)]
            up.append({'cluster': c, 'method': m, 'expressed_genes': len(v),
                       'genes_up': np.mean(v > (0.5 if m == 'wilcoxon' else 0))})
        ratio = np.median(depth[own]) / np.median(depth[~own])
        clusters.append({'cluster': c, 'name': obs.loc[own, 'SubAssignment'].iloc[0], 'cells': int(own.sum()),
                         'median_depth': np.median(depth[own]), 'depth_ratio': ratio})
        print(f'  {c}: depth ratio {ratio:.2f}', flush=True)
    ranks = pd.concat(rows, ignore_index=True)
    sets = {}
    for line in open(GMT):
        p = line.rstrip('\n').split('\t')
        sets[p[0]] = [g for g in p[2:] if g]
    focus = genes.get_indexer([g for g in sets[FOCUS_SET] if g in genes])
    focus = focus[focus >= 0]
    cp = sp.csr_matrix(sp.diags(1e4 / depth) @ x)[:, focus]
    own = lab == FOCUS
    detail = pd.DataFrame({'gene': genes[focus],
                           'detection_cluster': np.asarray((cp[own] > 0).mean(0)).ravel(),
                           'detection_rest': np.asarray((cp[~own] > 0).mean(0)).ravel(),
                           'mean_cluster': np.asarray(cp[own].mean(0)).ravel(),
                           'mean_rest': np.asarray(cp[~own].mean(0)).ravel()})
    depths = pd.DataFrame({'cluster': lab, 'depth': depth}).sort_values(['cluster', 'depth'], ignore_index=True)
    return pd.DataFrame(clusters).sort_values('cluster', ignore_index=True), \
        ranks.sort_values(['cluster', 'method', 'gene'], ignore_index=True), \
        pd.DataFrame(up).sort_values(['cluster', 'method'], ignore_index=True), \
        detail.sort_values('gene', ignore_index=True), depths


def variance(x, lab):
    """Each cluster's variance ratio at equal mean against the other myeloid cells, on CP10K, as clustering.multires_gaps
    computes Fig 2e's: 2 to the median |log2 variance ratio| over the genes detected in at least MIN_DETECT cells on
    each side whose means agree within EQUAL_MEAN."""
    depth = np.asarray(x.sum(1)).ravel()
    mom = mg.moments(sp.csr_matrix(sp.diags(1e4 / depth) @ x), lab, CLUSTERS)
    rows = [{'cluster': c, **{k: v for k, v in mg.gaps(*mom[c]).items() if k in ('gap_equal_mean', 'genes_equal_mean')}}
            for c in CLUSTERS]
    out = pd.DataFrame(rows)
    out['variance_ratio'] = 2 ** out['gap_equal_mean']
    return out


def draw_variance_vs_depth(ax, clusters, var):
    """Each cluster's variance ratio at equal mean against its |log2 depth ratio|, as clustering.multires_gaps'
    draw_gap_vs_depth, in its colour (as draw_depth_ratio's bars name them), with its least-squares line and the
    Spearman correlation."""
    d = clusters.merge(var, on='cluster')
    x = np.abs(np.log2(d['depth_ratio']))
    fit = linregress(x, d['variance_ratio'])
    span = np.array([0, x.max() * 1.05])
    ax.plot(span, fit.intercept + fit.slope * span, color='0.4', lw=0.9, ls='--', zorder=2)
    ax.scatter(x, d['variance_ratio'], s=16, c=[cluster_colour(c) for c in d['cluster']], lw=0, zorder=3)
    rho = spearmanr(x, d['variance_ratio']).statistic
    ax.text(0.04, 0.95, f'$\\rho$ = {rho:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=plot_style.SMALL)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1)
    mg.ratio_ticks(ax)
    ax.set_xlabel('|log2 depth ratio|')
    ax.set_ylabel(mg.RATIO['gap_equal_mean'])


STORY = ('Myc Targets V1', 'E2F Targets', 'Interferon Gamma Response')   # the sets of the authors' MC3 story
VOLCANO = {'wilcoxon': ('wilcoxon', 'wilcoxon_z', 'AUC', 0.5), 'ln': ('ln_lfc', 'ln', 'LFC', 0.0)}   # effect, statistic
LABELLED = 2   # the STORY genes with the strongest opposite calls (up under the Wilcoxon, down under LN), named


def story_genes():
    """The genes of STORY's sets, from GMT (Enrichr's copy of Hallmark, as the focus set's)."""
    genes = set()
    for line in open(GMT):
        p = line.rstrip('\n').split('\t')
        if p[0] in STORY:
            genes |= {g for g in p[2:] if g}
    return genes


def volcano_table(ranks, cluster=FOCUS):
    """The cluster's expressed genes with each method's effect and two-sided -log10 p (from the log survival function, so
    nothing underflows), and whether each is a STORY gene."""
    d = ranks[(ranks['cluster'] == cluster) & ranks['expressed']].pivot(index='gene', columns='method', values='value')
    out = pd.DataFrame(index=d.index)
    for m, (effect, stat, _, _) in VOLCANO.items():
        out[f'{m}_effect'] = d[effect]
        out[f'{m}_mlog10p'] = -(np.log(2) + norm.logsf(np.abs(d[stat].astype(float)))) / np.log(10)
    out['story'] = out.index.isin(story_genes())
    return out.dropna()


def opposite(table):
    """The STORY genes up under the Wilcoxon and down under LN, strongest opposite calls first (the smaller of the two
    -log10 p)."""
    d = table[table['story'] & (table['wilcoxon_effect'] > 0.5) & (table['ln_effect'] < 0)]
    return d.assign(both=d[['wilcoxon_mlog10p', 'ln_mlog10p']].min(axis=1)).sort_values('both', ascending=False)


def draw_volcano(ax, table, method):
    """One method's volcano for the cluster, -log10 p on a log axis: every expressed gene grey, STORY's genes in the
    method's colour, the LABELLED strongest opposite calls named; the shares of all genes and of STORY's genes up."""
    effect, stat, xlabel, ref = VOLCANO[method]
    x, y = table[f'{method}_effect'], table[f'{method}_mlog10p'].clip(lower=0.1)
    ax.scatter(x[~table['story']], y[~table['story']], s=3, lw=0, color='0.78', rasterized=True)
    ax.scatter(x[table['story']], y[table['story']], s=6, lw=0, color=method_color(COLOUR[method]), rasterized=True)
    right = method == 'wilcoxon'   # the opposite calls sit on the Wilcoxon's right arm, on LN's left one
    for k, g in enumerate(opposite(table).index[:LABELLED]):
        ax.annotate(g, (x[g], y[g]), xytext=(18, -3 - 12 * k) if right else (-6, 6 - 12 * k), textcoords='offset points',
                    ha='left' if right else 'right', va='center', fontsize=plot_style.SMALL,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='0.2', shrinkA=0, shrinkB=1))
    up = lambda v: f'{np.mean(v > ref):.0%}'
    ax.text(0.03, 0.97, f'genes up: all {up(x)}\nMYC, E2F, IFN-γ {up(x[table["story"]])}', transform=ax.transAxes,
            ha='left', va='top', fontsize=plot_style.SMALL, linespacing=1.2,
            bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.85), zorder=4)
    ax.axvline(ref, color='0.5', lw=0.6, ls=':')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 3e5)   # headroom above the points for the shares
    ax.set_yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('−log10 p')


CALLS = {'same': ('Same call', '0.55'), 'lost': ('Not significant under LN', '#fdbb84'),
         'reversed': ('Reversed under LN', '#d62728'), 'gained': ('Significant under LN only', method_color('ln')),
         'neither': ('Neither significant', '0.9')}   # each cluster's Fig. 4d pathways, the authors' procedure against LN


def call_changes(gsea, rows=ROWS, comp='wilcoxon'):
    """Per cluster, how many of rows' Hallmark sets keep their call (FDR < Q) under LN, lose it, reverse it, or gain one."""
    w = gsea[gsea['method'] == comp].set_index(['cluster', 'pathway'])
    l = gsea[gsea['method'] == 'ln'].set_index(['cluster', 'pathway'])
    d = w[['nes', 'fdr']].join(l[['nes', 'fdr']], lsuffix='_w', rsuffix='_ln')
    d = d[d.index.get_level_values('pathway').isin(rows)]
    sw, sl = d['fdr_w'] < Q, d['fdr_ln'] < Q
    same = np.sign(d['nes_w']) == np.sign(d['nes_ln'])
    d['call'] = np.select([sw & sl & same, sw & sl & ~same, sw & ~sl, ~sw & sl], ['same', 'reversed', 'lost', 'gained'],
                          'neither')
    t = d.groupby([d.index.get_level_values('cluster'), 'call']).size().unstack(fill_value=0)
    return t.reindex(index=CLUSTERS, columns=list(CALLS), fill_value=0)


def draw_call_changes(ax, gsea, comp='wilcoxon'):
    """call_changes as one vertical bar per cluster, MC1 to MC9 as the heatmaps' columns, segments in CALLS' order from
    the bottom."""
    t = call_changes(gsea, comp=comp)
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for k, (label, colour) in CALLS.items():
        ax.bar(x, t[k], bottom=bottom, width=0.75, color=colour, label=label)
        bottom += t[k].to_numpy()
    ax.set_xticks(x, [LABEL[c] for c in t.index], rotation=90)
    ax.set_xlim(-0.5, len(t) - 0.5)
    ax.set_ylim(0, bottom.max())
    ax.set_yticks(range(0, int(bottom.max()) + 1, 5))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Hallmark gene sets')
    ax.grid(False, axis='x')


def draw_genes_up(ax, up, clusters, methods=('wilcoxon', 'ln')):
    """Each cluster's share of its expressed genes ranked up against its log2 depth ratio, one colour per method."""
    d = up.merge(clusters[['cluster', 'depth_ratio']], on='cluster')
    for m in methods:
        e = d[d['method'] == m].assign(x=lambda t: np.log2(t['depth_ratio'])).sort_values('x')
        ax.plot(e['x'], e['genes_up'], marker='o', ms=3, lw=1.0, color=method_color(COLOUR[m]), label=NAME[m])
    ax.axhline(0.5, color='0.5', lw=0.6, ls=':')
    ax.set_ylim(0, 1)
    ax.set_xlabel('log2 depth ratio')
    ax.set_ylabel('Expressed genes ranked up')


def levels(comparator):
    """A figure's two levels, the comparator's values and LN's: per gene (GENE) and per set (NES)."""
    return {'genes': (GENE[comparator], GENE['ln']), 'pathways': ((comparator, 'NES', 0.0), ('ln', 'NES', 0.0))}


def values_by_cluster(ranks, gsea):
    """Per level and method, each cluster's values: its expressed genes' value (GENE), its Hallmark sets' NES."""
    out = {}
    for m, _, _ in GENE.values():
        d = ranks[(ranks['method'] == m) & ranks['expressed'] & np.isfinite(ranks['value'])]
        out['genes', m] = {c: g['value'].to_numpy() for c, g in d.groupby('cluster')}
    for m in (*COMPARATORS, 'ln'):
        d = gsea[(gsea['method'] == m) & np.isfinite(gsea['nes'])]
        out['pathways', m] = {c: g['nes'].to_numpy() for c, g in d.groupby('cluster')}
    return out


def readouts(clusters, gsea, up, values):
    """Per cluster and method: agreement with the published signs, the share of sets with positive NES, the sets
    called, the share of expressed genes ranked up, and the median of each level's values; per method and level, the
    Pearson correlation with the log2 depth ratio over the clusters."""
    rows = []
    for (c, m), o in gsea.groupby(['cluster', 'method']):
        o = o.set_index('pathway')
        sign = np.array([1 if PUBLISHED[p][CLUSTERS.index(c)] == '+' else -1 for p in PUBLISHED])
        rows.append({'cluster': c, 'method': m,
                     'published_sign_agreement': np.mean(np.sign(o.loc[list(PUBLISHED), 'nes']) == sign),
                     'sets_positive': np.mean(o['nes'] > 0), 'median_nes': np.nanmedian(o['nes']),
                     'nes_undefined': int(o['nes'].isna().sum()),
                     'calls': int((o['fdr'] < Q).sum())})
    out = (pd.DataFrame(rows).merge(up[['cluster', 'method', 'genes_up']], on=['cluster', 'method'])
           .merge(clusters[['cluster', 'depth_ratio']], on='cluster'))
    x = np.log2(clusters.set_index('cluster').loc[CLUSTERS, 'depth_ratio'].to_numpy())
    corr = []
    for (level, m), v in values.items():
        med = np.array([np.median(v[c]) for c in CLUSTERS])
        corr.append({'level': level, 'method': m, 'summary': 'median', 'pearson': np.corrcoef(x, med)[0, 1]})
    for m, d in out.groupby('method'):
        d = d.set_index('cluster').loc[CLUSTERS]
        for col in ('genes_up', 'sets_positive'):
            corr.append({'level': col, 'method': m, 'summary': 'share', 'pearson': np.corrcoef(x, d[col])[0, 1]})
    return out, pd.DataFrame(corr)


def cluster_colour(c):
    """A cluster's colour, as the UMAP draws it."""
    return plt.get_cmap('tab10')(CLUSTERS.index(c))


def contrast_limit(gsea, methods):
    """The y limit shared by the contrast's facets: the largest |NES| of MC3 and MC4 in CONTRAST_SETS, any of methods."""
    g = gsea[gsea['cluster'].isin(CONTRAST) & gsea['pathway'].isin(list(CONTRAST_SETS)) & gsea['method'].isin(methods)]
    return float(np.ceil(g['nes'].abs().max() * 1.1))


def draw_contrast(ax, gsea, method, ylabel=True, lim=None):
    """MC3's and MC4's NES for CONTRAST_SETS under one method, as paired vertical bars in the clusters' colours, on a
    symmetric y range (lim, or the method's own)."""
    g = gsea[gsea['method'] == method].set_index(['cluster', 'pathway'])['nes']
    sets = list(CONTRAST_SETS)
    x = np.arange(len(sets))
    w = 0.38
    for k, c in enumerate(CONTRAST):
        ax.bar(x + (k - 0.5) * w, [g[(c, p)] for p in sets], width=w, color=cluster_colour(c), label=LABEL[c])
    ax.axhline(0, color='0.3', lw=0.6)
    if lim is None:
        lim = np.ceil(np.max(np.abs([g[(c, p)] for c in CONTRAST for p in sets])) * 1.1)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONTRAST_SETS[p] for p in sets], rotation=90)
    if ylabel:
        ax.set_ylabel('NES')
    else:
        ax.tick_params(labelleft=False)
    ax.grid(False)


def draw_contrast_cluster(ax, gsea, cluster, methods, ylabel=True, lim=None):
    """One cluster's NES for CONTRAST_SETS under each method, as side-by-side vertical bars in the methods' colours, on a
    symmetric y range (lim, or the cluster's own)."""
    g = gsea[gsea['cluster'] == cluster].set_index(['method', 'pathway'])['nes']
    sets = list(CONTRAST_SETS)
    x = np.arange(len(sets))
    w = 0.38
    for k, m in enumerate(methods):
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, [g[(m, p)] for p in sets], width=w, color=method_color(COLOUR[m]),
               label=TITLE[m])
    ax.axhline(0, color='0.3', lw=0.6)
    if lim is None:
        lim = np.ceil(np.max(np.abs([g[(m, p)] for m in methods for p in sets])) * 1.1)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONTRAST_SETS[p] for p in sets], rotation=90)
    if ylabel:
        ax.set_ylabel('NES')
    else:
        ax.tick_params(labelleft=False)
    ax.grid(False)


def order_of(clusters):
    """The clusters in the paper's order, MC1 to MC9, with their labels."""
    order = [c for c in CLUSTERS if c in set(clusters['cluster'])]
    return order, [LABEL[k] for k in order]


def grids(gsea, order, methods):
    """The NES of the Fig. 4d sets for the heatmaps, keyed by method."""
    return {m: gsea[gsea['method'] == m].pivot(index='pathway', columns='cluster', values='nes')
            .reindex(index=ROWS, columns=order) for m in methods}


def draw_heatmap(ax, nes, ticks, row_labels=True, vmax=SHARED_VMAX):
    """One heatmap of NES, pathways by clusters, on a symmetric colour scale (values beyond it saturate); returns the
    image for a colour bar."""
    v = vmax
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.8')         # a set whose NES fgsea cannot normalise (no random set of its sign)
    im = ax.imshow(nes.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v), aspect='auto')
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([SHORT.get(r, r) for r in ROWS] if row_labels else [])
    ax.set_xlabel('Cluster')
    ax.grid(False)
    return im


def draw_colorbar(cax, im):
    cb = plt.colorbar(im, cax=cax, extend='both')
    cb.set_label('NES')


def draw_depths(ax, depths, clusters):
    """Counts per cell in each cluster, MC1 at the top."""
    order, _ = order_of(clusters)
    order = order[::-1]
    ax.boxplot([np.log10(depths.loc[depths['cluster'] == c, 'depth']) for c in order], vert=False, showfliers=False,
               widths=0.6, medianprops={'color': 'k', 'lw': 1})
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([LABEL[c] for c in order])
    ax.set_xlabel('log10 total counts')


def draw_depth_ratio(ax, clusters):
    """Each cluster's median counts per cell over the other myeloid cells', log2, as vertical bars, MC1 to MC9."""
    order, labels = order_of(clusters)
    r = np.log2(clusters.set_index('cluster').loc[order, 'depth_ratio'].to_numpy())
    x = np.arange(len(order))
    ax.bar(x, r, width=0.7, color=[cluster_colour(c) for c in order])
    ax.axhline(0, color='0.3', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel('log2 depth ratio')
    ax.grid(False)


CALL_TESTS = ('wilcoxon', 'ln')


def draw_calls(ax, calls):
    """Each cluster's significant genes under each test, as side-by-side vertical bars, MC1 to MC9, as
    draw_depth_ratio's."""
    d = calls.set_index(['cluster', 'method'])['calls']
    order, labels = order_of(calls.drop_duplicates('cluster'))
    x = np.arange(len(order))
    w = 0.38
    for k, m in enumerate(CALL_TESTS):
        ax.bar(x + (k - 0.5) * w, [d[(c, m)] for c in order], width=w, color=method_color(COLOUR[m]), label=NAME[m])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel('DEGs')
    plot_style.boxplot_grid(ax)


def draw_calls_vs_cells(ax, calls):
    """Each cluster's DEGs under each test against its number of cells, on a log axis, with each test's least-squares
    line (on log cells) and Spearman correlation in its colour."""
    for k, m in enumerate(CALL_TESTS):
        d = calls[calls['method'] == m]
        x, y = np.log10(d['cells']), d['calls']
        fit = linregress(x, y)
        span = np.array([x.min(), x.max()])
        ax.plot(10 ** span, fit.intercept + fit.slope * span, color=method_color(COLOUR[m]), lw=0.9, ls='--', zorder=2)
        ax.scatter(d['cells'], y, s=14, color=method_color(COLOUR[m]), lw=0, zorder=3, label=NAME[m])
        ax.text(0.97, 0.04 + 0.09 * k, f'$\\rho$ = {spearmanr(x, y).statistic:.2f}', color=method_color(COLOUR[m]),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=plot_style.SMALL)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Cells in the cluster')
    ax.set_ylabel('DEGs')


def draw_calls_vs_depth(ax, calls):
    """Each cluster's DEGs under the Wilcoxon over LN's against its |log2 depth ratio|, as draw_variance_vs_depth, in
    its colour, with its least-squares line and the Spearman correlation."""
    d = calls.pivot_table(index=['cluster', 'depth_ratio'], columns='method', values='calls').reset_index()
    x, y = np.abs(np.log2(d['depth_ratio'])), d['wilcoxon'] / d['ln']
    fit = linregress(x, y)
    span = np.array([0, x.max() * 1.05])
    ax.plot(span, fit.intercept + fit.slope * span, color='0.4', lw=0.9, ls='--', zorder=2)
    ax.scatter(x, y, s=16, c=[cluster_colour(c) for c in d['cluster']], lw=0, zorder=3)
    ax.text(0.04, 0.95, f'$\\rho$ = {spearmanr(x, y).statistic:.2f}', transform=ax.transAxes, ha='left', va='top',
            fontsize=plot_style.SMALL)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1)
    ax.set_xlabel('|log2 depth ratio|')
    ax.set_ylabel(f"{NAME['wilcoxon']} DEGs / {NAME['ln']} DEGs")


def draw_umap(ax, umap, by, cax=None, cbar_label=True):
    """The myeloid cells' UMAP coloured by cluster (labelled at each cluster's median) or by log10 counts per cell, the
    colour scale between its 1st and 99th percentiles."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    order = np.random.default_rng(0).permutation(len(umap))
    u = umap.iloc[order]
    if by == 'cluster':
        colours = {c: cluster_colour(c) for c in CLUSTERS}
        ax.scatter(u['umap_1'], u['umap_2'], s=0.3, lw=0, c=[colours[c] for c in u['cluster']], rasterized=True)
        for c, g in umap.groupby('cluster'):
            ax.text(g['umap_1'].median(), g['umap_2'].median(), LABEL[c], fontsize=plot_style.SMALL,
                    ha='center', va='center', weight='bold')
    else:
        d = np.log10(u['depth'])
        im = ax.scatter(u['umap_1'], u['umap_2'], s=0.3, lw=0, c=d, cmap='viridis', rasterized=True,
                        vmin=np.percentile(d, 1), vmax=np.percentile(d, 99))
        cb = plt.colorbar(im, cax=cax) if cax is not None else plt.colorbar(im, ax=ax)
        if cbar_label:
            cb.set_label('log10 total counts')


def spread(x, gap):
    """Positions for boxes at x, each at least gap right of the one before it (closer ones nudged right)."""
    p = np.array(x, dtype=float)
    order = np.argsort(p, kind='stable')
    for a, b in zip(order[:-1], order[1:]):
        p[b] = max(p[b], p[a] + gap)
    return p


def value_range(values):
    """The lowest and highest value over a level's clusters and methods, padded, so every point drawn shows."""
    lo = min(d.min() for v in values for d in v.values())
    hi = max(d.max() for v in values for d in v.values())
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def whisker_range(values):
    """The lowest and highest whisker over a level's clusters and methods, padded, for boxes drawn without their points."""
    stats = [s for v in values for s in cbook.boxplot_stats(list(v.values()))]
    lo, hi = min(s['whislo'] for s in stats), max(s['whishi'] for s in stats)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def draw_by_depth(ax, values, clusters, method, ylabel, ref, width=0.07, ylim=None, show_ylabel=True, points=True):
    """Each cluster's values (its genes or its sets) as a box at its log2 depth ratio (boxes closer than width nudged
    right), every value a point over its box unless points is False."""
    x = np.log2(clusters.set_index('cluster').loc[CLUSTERS, 'depth_ratio'].to_numpy())
    data = [values[c] for c in CLUSTERS]
    positions = spread(x, width * 1.15)
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False, patch_artist=True,
                    manage_ticks=False, medianprops={'color': 'k', 'lw': 0.8}, whiskerprops={'lw': 0.6},
                    capprops={'lw': 0.6}, boxprops={'lw': 0.6})
    for b in bp['boxes']:
        b.set_facecolor(method_color(COLOUR[method]))
    if points:
        rng = np.random.default_rng(0)
        for pos, d in zip(positions, data):
            plot_style.strip(ax, pos, d, method_color(COLOUR[method]), width, rng)
    ax.axhline(ref, color='0.6', lw=0.6, ls=':')
    ax.set_xlabel('log2 depth ratio')
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)


def draw_focus(ax, detail, gsea):
    """The focus set's genes in the focus cluster: detection (cluster minus rest) against the log2 ratio of means,
    genes detected in more than 5% of cells; the share detected more at a lower mean and each method's NES."""
    d = detail[(detail['detection_cluster'] + detail['detection_rest']) / 2 > 0.05]
    dx = 100 * (d['detection_cluster'] - d['detection_rest'])
    dy = np.log2(d['mean_cluster'] / d['mean_rest'])
    ax.scatter(dx, dy, s=4, lw=0, color='0.3')
    ax.axhline(0, color='0.5', lw=0.6)
    ax.axvline(0, color='0.5', lw=0.6)
    o = gsea[(gsea['cluster'] == FOCUS) & (gsea['pathway'] == FOCUS_SET)].set_index('method')
    ax.text(0.98, 0.02, f'{np.mean((dx > 0) & (dy < 0)):.0%} detected more,\nat a lower mean\n'
            f'NES: {NAME["wilcoxon"]} {o.loc["wilcoxon", "nes"]:+.2f}\n{NAME["ln"]} {o.loc["ln", "nes"]:+.2f}',
            transform=ax.transAxes, fontsize=plot_style.SMALL, ha='right', va='bottom')
    ax.set_xlabel('Detection difference (% of cells)')
    ax.set_ylabel('Mean, log2 ratio')


def plot(clusters, gsea, detail, depths, values, umap):
    """The arm's figures, one plot each: the two heatmaps, the depths and depth ratios, the genes ranked up and the
    sets with positive NES against depth, the focus cluster's genes, the UMAP by cluster and by depth."""
    plot_style.use()
    order, ticks = order_of(clusters)
    for m, nes in grids(gsea, order, (*COMPARATORS, 'ln')).items():
        fig = plt.figure(figsize=(4.2, 3.4))
        ax = fig.add_axes([0.42, 0.25, 0.46, 0.66])
        draw_colorbar(fig.add_axes([0.9, 0.25, 0.02, 0.66]), draw_heatmap(ax, nes, ticks))
        ax.set_title(TITLE[m])
        save_plot(fig, f'abdelfattah_heatmap_{m}', key=False)
    for stem, draw, key in (('depths', lambda ax: draw_depths(ax, depths, clusters), False),
                            ('depth_ratio', lambda ax: draw_depth_ratio(ax, clusters), False),
                            ('umap_cluster', lambda ax: draw_umap(ax, umap, 'cluster'), False),
                            ('umap_depth', lambda ax: draw_umap(ax, umap, 'depth'), False),
                            ('focus', lambda ax: draw_focus(ax, detail, gsea), False)):
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw(ax)
        save_plot(fig, f'abdelfattah_{stem}', key=key)
    for comp in COMPARATORS:
        pair = (comp, 'ln')
        fig, axes = plt.subplots(1, 2, figsize=(4.0, 2.4))
        for k, (ax, m) in enumerate(zip(axes, pair)):
            draw_contrast(ax, gsea, m, ylabel=k == 0, lim=contrast_limit(gsea, pair))
            ax.set_title(TITLE[m])
        axes[-1].legend(loc='upper right', fontsize=plot_style.SMALL, frameon=False)
        save_plot(fig, f'abdelfattah_contrast_{comp}', key=False)
        for level, entries in levels(comp).items():
            shared = value_range([values[level, m] for m, _, _ in entries]) if level == 'pathways' else None
            for k, (m, ylabel, ref) in enumerate(entries):
                fig, ax = plt.subplots(figsize=plot_style.PANEL)
                draw_by_depth(ax, values[level, m], clusters, m, ylabel, ref, ylim=shared,
                              show_ylabel=shared is None or k == 0, points=level == 'pathways')
                ax.set_title(NAME[pair[k]])
                save_plot(fig, f'abdelfattah_{comp}_{level}_{m}', key=False)


def save_plot(fig, stem, key=True):
    if key:
        plot_style.legend_outside(fig)
    path = OUT / fig_name(stem)
    plot_style.rasterize_dense(fig)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def tables(sources=None):
    """The saved tables the readouts and plots read, with the sets under the figures' names."""
    sources = sources or {}
    t = {k: pd.read_csv(require_input(OUT / f'{k}.csv', what=f'the {k} table',
                                      source=sources.get(k, 'python -m published.abdelfattah2022 --rankings')))
         for k in ('clusters', 'rankings', 'genes_up', 'focus_genes', 'depths', 'gsea', 'umap')}
    t['gsea']['pathway'] = t['gsea']['pathway'].map(lambda p: DISPLAY.get(p, p))
    return t


def calls():
    """Per cluster against the rest, each test's Bonferroni calls (adjusted p below Q) and the share of them up: LN's
    t-test as the package runs it, the Wilcoxon as scanpy runs it (tie-corrected), with the cluster's size and depth."""
    x, genes, obs = load()
    lab = obs['SubCluster'].astype(str).to_numpy()
    a, cats = normalised(x, lab, genes)
    rank_genes_groups_ln(a, 'g', sparse=True, key_added='ln', layer='norm_counts', corr_method='bonferroni')
    sc.tl.rank_genes_groups(a, 'g', method='wilcoxon', layer='log1p_norm', use_raw=False, tie_correct=True,
                            corr_method='bonferroni', key_added='wilcoxon')
    depth = np.asarray(x.sum(1)).ravel()
    rows = []
    for c in cats:
        own = lab == c
        for m in ('ln', 'wilcoxon'):
            d = _get_rank_genes_df(a, c, m)
            sig = d['pvals_adj'] < Q
            rows.append({'cluster': c, 'method': m, 'cells': int(own.sum()),
                         'depth_ratio': np.median(depth[own]) / np.median(depth[~own]), 'genes': len(d),
                         'calls': int(sig.sum()), 'up': int((sig & (d['scores'] > 0)).sum())})
    return pd.DataFrame(rows).assign(share_up=lambda t: t['up'] / t['calls'])


# The paper's text on its Fig. 4d, as claims about NES: (cluster, set) cells that should be up (FDR < Q and NES > 0),
# should not be, or MC04 below another cluster.
INFLAMMATORY = ['Interferon Alpha Response', 'TNF-alpha Signaling via NF-kB', 'Allograft Rejection', 'Inflammatory Response']
PROLIFERATION = ['Myc Targets V1', 'Oxidative Phosphorylation', 'Interferon Gamma Response', 'G2-M Checkpoint', 'E2F Targets']
CLAIMS = {
    'hypoxia_emt': ('Hypoxia and EMT enriched in MC3-MC5', 'up',
                    [(c, p) for c in ('MC03', 'MC04', 'MC05') for p in ('Hypoxia', 'Epithelial Mesenchymal Transition')]),
    'mdsc_lower': ('MDSCs (MC4) lower than the macrophages in MYC targets, OXPHOS, IFN-gamma and proliferation', 'below',
                   [(c, p) for c in ('MC03', 'MC05', 'MC09') for p in PROLIFERATION]),
    'inflammatory_in': ('Inflammatory hallmarks enriched in MC3, MC4, MC5, MC6 and MC8', 'up',
                        [(c, p) for c in ('MC03', 'MC04', 'MC05', 'MC06', 'MC08') for p in INFLAMMATORY]),
    'inflammatory_not': ('Inflammatory hallmarks not enriched in MC2 and MC7', 'not_up',
                         [(c, p) for c in ('MC02', 'MC07') for p in INFLAMMATORY]),
    'oxphos': ('OXPHOS enriched in MC3, MC5, MC6, MC8 and MC9', 'up',
               [(c, 'Oxidative Phosphorylation') for c in ('MC03', 'MC05', 'MC06', 'MC08', 'MC09')]),
}
CLAIM_METHODS = ('wilcoxon', 'wilcoxon_z', 'ln')


def claims(gsea, methods=CLAIM_METHODS):
    """Each claim's cells under each method: NES, FDR and whether the cell agrees with the claim ('below': MC04's NES
    under the cell's cluster's)."""
    g = gsea.set_index(['method', 'cluster', 'pathway'])
    rows = []
    for key, (text, kind, cells) in CLAIMS.items():
        for m in methods:
            for c, p in cells:
                nes, fdr = g.loc[(m, c, p), 'nes'], g.loc[(m, c, p), 'fdr']
                up = nes > 0 and fdr < Q
                holds = {'up': up, 'not_up': not up, 'below': g.loc[(m, 'MC04', p), 'nes'] < nes}[kind]
                rows.append({'claim': key, 'text': text, 'method': m, 'cluster': c, 'pathway': p, 'nes': nes, 'fdr': fdr,
                             'holds': holds})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Abdelfattah 2022 Fig. 4d under the authors\' procedure and LN')
    parser.add_argument('--rankings', action='store_true',
                        help='Step 1: load the data and write the rankings and per-cluster tables')
    parser.add_argument('--variance', action='store_true',
                        help="Each cluster's variance ratio at equal mean (variance.csv) and its plot against depth")
    parser.add_argument('--calls', action='store_true',
                        help="Each test's Bonferroni calls per cluster and the share up, LN and the Wilcoxon (calls.csv)")
    parser.add_argument('--plot-only', action='store_true', help='With --calls: redraw from calls.csv')
    parser.add_argument('--claims', action='store_true',
                        help="The paper's Fig. 4d claims under the authors' procedure, the Wilcoxon z and LN (claims.csv)")
    args = parser.parse_args()
    if args.calls:
        if args.plot_only:
            table = pd.read_csv(require_input(OUT / 'calls.csv', what='the calls table', source='--calls'))
        else:
            table = calls()
            table.to_csv(OUT / 'calls.csv', index=False)
            print(f'  wrote {OUT / "calls.csv"}')
            print(table.round(3).to_string(index=False))
        plot_style.use()
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_calls(ax, table)
        save_plot(fig, 'abdelfattah_calls')
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_calls_vs_cells(ax, table)
        save_plot(fig, 'abdelfattah_calls_vs_cells')
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_calls_vs_depth(ax, table)
        save_plot(fig, 'abdelfattah_calls_vs_depth', key=False)
        return
    if args.claims:
        gsea = tables({'gsea': 'Rscript published/R/abdelfattah2022_fgsea.R'})['gsea']
        table = claims(gsea)
        table.to_csv(OUT / 'claims.csv', index=False)
        print(f'  wrote {OUT / "claims.csv"}')
        print(table.groupby(['claim', 'method'], sort=False)['holds'].agg(['sum', 'size']).unstack('method').to_string())
        return
    if args.variance:
        x, _, obs = load()
        var = variance(x, obs['SubCluster'].astype(str).to_numpy())
        var.to_csv(OUT / 'variance.csv', index=False)
        print(f'  wrote {OUT / "variance.csv"}')
        print(var.round(3).to_string(index=False))
        plot_style.use()
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        clusters = pd.read_csv(require_input(OUT / 'clusters.csv', what='the clusters table', source='--rankings'))
        draw_variance_vs_depth(ax, clusters, var)
        save_plot(fig, 'abdelfattah_variance_vs_depth', key=False)
        return
    if args.rankings:
        clusters, ranks, up, detail, depths = run()
        for k, v in {'clusters': clusters, 'rankings': ranks, 'genes_up': up, 'focus_genes': detail,
                     'depths': depths}.items():
            v.to_csv(OUT / f'{k}.csv', index=False)
            print(f'  wrote {OUT / f"{k}.csv"}')
        return
    sources = {'gsea': 'Rscript published/R/abdelfattah2022_fgsea.R', 'umap': 'Rscript published/R/abdelfattah2022_umap.R'}
    t = tables(sources)
    values = values_by_cluster(t['rankings'], t['gsea'])
    summary, corr = readouts(t['clusters'], t['gsea'], t['genes_up'], values)
    for k, v in {'summary': summary, 'depth_correlations': corr}.items():
        v.to_csv(OUT / f'{k}.csv', index=False)
        print(f'  wrote {OUT / f"{k}.csv"}')
    plot(t['clusters'], t['gsea'], t['focus_genes'], t['depths'], values, t['umap'])
    table = volcano_table(t['rankings'])
    for m in VOLCANO:
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_volcano(ax, table, m)
        ax.set_title(f'{LABEL[FOCUS]}, {NAME[m]}')
        save_plot(fig, f'abdelfattah_volcano_{m}', key=False)
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_call_changes(ax, t['gsea'])
    ax.legend(frameon=False, fontsize=plot_style.SMALL, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save_plot(fig, 'abdelfattah_call_changes', key=False)
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_genes_up(ax, t['genes_up'], t['clusters'])
    ax.legend(frameon=False, fontsize=plot_style.SMALL)
    save_plot(fig, 'abdelfattah_genes_up', key=False)


if __name__ == '__main__':
    main()
