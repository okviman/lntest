#!/usr/bin/env python3
"""Is LN's SE calibrated on real data, does it say which fold changes carry over to another dataset, and what does its
decomposition of the fold change show?

PBMC3k and PBMC10k as clustering/replication.py loads them, each of its 6 types against the other included cells. LN
is get_LN_lfcs_sparse on CP10K as rank_genes_groups_ln calls it; the log1p t-test's effect is the difference in mean
log1p CP10K with Welch's SE, the quantities its t divides. 1, split halves: SPLITS random halvings of every type's cells,
z = (effect_A - effect_B) / sqrt(SE_A^2 + SE_B^2) per gene, type and split, by the number of cells detecting the gene
in the sparser group of the full data; a calibrated SE gives |z| < 1.96 in 95% of pairs. 2, across datasets: LN's
|LFC_10k - LFC_3k| and the correlation of the LFCs in quintiles of the combined SE, with scanpy's LFC in the same bins.
3, decomposition: LN's LFC is the difference in log2 detection rate plus the difference in log2 mean among detecting
cells, and its SE^2 a detection and a level term per group; LN's significant up genes classed by the detection part's
share. 4, CD14 Mono in PBMC3k as the picture. For the figures: the correlation between the datasets of each effect on
the genes all three methods call up in both; a forest of the example comparison's most extreme LFC per stratum; six
markers' parts in both datasets; and LN's top markers by its score and by its marker rule, its significant up genes
ranked by the detection score (the detection part over its SE).

    python -m clustering.lfc_se               # from reproducibility/; a few minutes
    python -m clustering.lfc_se --plot-only   # redraw from the CSVs
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import norm, pearsonr, spearmanr

import plot_style
from clustering import replication as rep
from evaluation_utils import _get_rank_genes_df
from lntest import get_LN_lfcs_sparse, rank_genes_groups_ln
from method_colors import method_color, method_label
from paths import fig_name, require_input, results_dir

OUT = results_dir(__file__) / 'lfc_se'
SPLITS = 10
STRATA = [(0, 0, '0'), (1, 4, '1-4'), (5, 19, '5-19'), (20, 99, '20-99'), (100, 499, '100-499'),
          (500, np.inf, '>=500')]
Z = norm.isf(0.025)
BINS = 5
HIST = np.arange(-8, 8.25, 0.25)
DETECTION, LEVEL = 2 / 3, 1 / 3
EXAMPLE = ('PBMC3k', 'CD14 Mono')
EFFECT = {'ln': "LN's LFC", 't_test': 'log1p t-test effect'}
LN2 = np.log(2) ** 2


def cp10k(x):
    """Rows rescaled to 10,000 in double precision, as LN normalises its input."""
    x = sp.csr_matrix(x, dtype=np.float64)
    s = np.asarray(x.sum(1)).ravel()
    return sp.diags(1e4 / s) @ x


def group_moments(x, own):
    """Cells, detecting cells, sum and sum of squares per gene in own and in the rest."""
    out = []
    for m in (own, ~own):
        sub = x[m]
        out.append((int(m.sum()), np.asarray((sub > 0).sum(0)).ravel().astype(float), np.asarray(sub.sum(0)).ravel(),
                    np.asarray(sub.multiply(sub).sum(0)).ravel()))
    return out


def ln_parts(x, own):
    """LN's LFC as detection part plus level part, and its SE^2 as detection and level terms of each group, from the
    positive values of the CP10K matrix x; a group where no cell detects the gene takes LN's fallback (one detecting
    cell, mean 1, level term 1)."""
    parts = {}
    for side, (n, a, s, ss) in zip(('own', 'rest'), group_moments(x, own)):
        zero = a == 0
        a_hat = np.where(zero, 1.0, a)
        m = np.where(zero, 1.0, s / np.maximum(a, 1))
        var = np.where(zero, 0.0, np.maximum(ss / np.maximum(a, 1) - m ** 2, 0))
        parts[side] = dict(det=np.log2(a_hat / n), lvl=np.log2(m), se_det=(1 / a_hat - 1 / n) / LN2,
                           se_lvl=np.where(zero, 1.0, np.log(1 + var / (a_hat * m ** 2))) / LN2, detecting=a)
    o, r = parts['own'], parts['rest']
    return pd.DataFrame({'det': o['det'] - r['det'], 'lvl': o['lvl'] - r['lvl'], 'se2_det_own': o['se_det'],
                         'se2_lvl_own': o['se_lvl'], 'se2_det_rest': r['se_det'], 'se2_lvl_rest': r['se_lvl'],
                         'detecting_own': o['detecting'], 'detecting_rest': r['detecting']})


def log1p_effect(x, own):
    """Difference in mean log1p and Welch's SE (variances with ddof 1)."""
    (n1, _, s1, q1), (n0, _, s0, q0) = group_moments(x, own)
    m1, m0 = s1 / n1, s0 / n0
    v1, v0 = (q1 - n1 * m1 ** 2) / (n1 - 1), (q0 - n0 * m0 ** 2) / (n0 - 1)
    return m1 - m0, np.sqrt(np.maximum(v1, 0) / n1 + np.maximum(v0, 0) / n0)


def ln_effect(x, own):
    lfc, _, se = get_LN_lfcs_sparse(x[own], x[~own], return_standard_error=True)
    return np.asarray(lfc, float), np.asarray(se, float)


def stratum(k):
    out = np.empty(len(k), dtype=object)
    for lo, hi, label in STRATA:
        out[(k >= lo) & (k <= hi)] = label
    return out


def full_data(name, adata, types):
    """Per type, every gene's LN LFC and SE (checked against rank_genes_groups_ln), its decomposition (checked against
    them), the log1p effect and SE (checked against scanpy's t), scanpy's LFC, LN's adjusted p and the sparser group's
    detecting cells; each method's score and adjusted p."""
    rep.run_tests(adata)
    cp, lg = adata.layers['norm_counts'], adata.layers['log1p_norm']
    x = cp10k(cp)
    rows = []
    for t in types:
        own = (adata.obs['type'] == t).to_numpy()
        lfc, se = ln_effect(cp, own)
        d, se_t = log1p_effect(lg, own)
        parts = ln_parts(x, own)
        ln = _get_rank_genes_df(adata, t, 'ln').set_index('names').reindex(adata.var_names)
        tt = _get_rank_genes_df(adata, t, 't_test').set_index('names').reindex(adata.var_names)
        wx = _get_rank_genes_df(adata, t, 'wilcoxon').set_index('names').reindex(adata.var_names)
        assert np.allclose(lfc, ln['logfoldchanges'], atol=1e-4), (name, t, 'LN LFC')
        assert np.allclose(lfc / se, ln['scores'], rtol=1e-3, atol=1e-3), (name, t, 'LN score')
        assert np.allclose(parts['det'] + parts['lvl'], lfc, atol=1e-6), (name, t, 'decomposition')
        se2 = parts[['se2_det_own', 'se2_lvl_own', 'se2_det_rest', 'se2_lvl_rest']].sum(axis=1)
        assert np.allclose(np.sqrt(se2), se, rtol=1e-6), (name, t, 'SE decomposition')
        ok = se_t > 0
        assert np.allclose((d / np.where(ok, se_t, 1))[ok], tt['scores'][ok], rtol=1e-3, atol=1e-2), (name, t, 't')
        rows.append(pd.DataFrame({'dataset': name, 'type': t, 'gene': adata.var_names, 'ln_lfc': lfc, 'ln_se': se,
                                  'ln_padj': ln['pvals_adj'].to_numpy(), 'log1p_effect': d, 'log1p_se': se_t,
                                  'scanpy_lfc': tt['logfoldchanges'].to_numpy(dtype=float),
                                  'sparser': np.minimum(parts['detecting_own'], parts['detecting_rest']),
                                  'ln_score': ln['scores'].to_numpy(dtype=float),
                                  't_padj': tt['pvals_adj'].to_numpy(), 't_score': tt['scores'].to_numpy(dtype=float),
                                  'w_padj': wx['pvals_adj'].to_numpy(), 'w_score': wx['scores'].to_numpy(dtype=float),
                                  **{c: parts[c].to_numpy() for c in parts}}))
    print(f'  {name}: LN, its decomposition and the log1p t reproduce the stored results', flush=True)
    return pd.concat(rows, ignore_index=True)


def split_halves(name, adata, types, full):
    """Readout 1: per split, type and gene, z for both methods, aggregated by stratum; and z histograms."""
    cp, lg = adata.layers['norm_counts'], adata.layers['log1p_norm']
    labels = adata.obs['type'].to_numpy()
    rows, hist = [], []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        half = np.zeros(adata.n_obs, dtype=bool)
        for t in types:
            idx = np.flatnonzero(labels == t)
            half[rng.permutation(idx)[:len(idx) // 2]] = True
        for t in types:
            f = full[full['type'] == t]
            keep = (f['detecting_own'] + f['detecting_rest'] > 0).to_numpy()
            strata = stratum(f['sparser'].to_numpy())
            eff = {}
            for h in (half, ~half):
                own = labels[h] == t
                eff.setdefault('ln', []).append(ln_effect(cp[h], own))
                eff.setdefault('t_test', []).append(log1p_effect(lg[h], own))
            for m, ((ea, sa), (eb, sb)) in eff.items():
                s = np.sqrt(sa ** 2 + sb ** 2)
                defined = keep & (s > 0) & np.isfinite(s)
                z = (ea - eb) / np.where(defined, s, 1)
                for label in (lab for _, _, lab in STRATA):
                    sel = defined & (strata == label)
                    rows.append({'dataset': name, 'method': m, 'stratum': label, 'type': t, 'split': seed,
                                 'pairs': int(sel.sum()), 'covered': int((np.abs(z[sel]) < Z).sum()),
                                 'sum_z2': float((z[sel] ** 2).sum()),
                                 'undefined': int((keep & ~defined & (strata == label)).sum())})
                    c, _ = np.histogram(np.clip(z[sel], HIST[0], HIST[-1] - 1e-9), bins=HIST)
                    hist.append(pd.DataFrame({'dataset': name, 'method': m, 'stratum': label, 'bin': HIST[:-1],
                                              'count': c}))
        print(f'  {name}: split {seed + 1}/{SPLITS}', flush=True)
    per = pd.DataFrame(rows)
    hist = pd.concat(hist).groupby(['dataset', 'method', 'stratum', 'bin'], sort=False)['count'].sum().reset_index()
    return per, hist


def calibration(per):
    g = per.groupby(['dataset', 'method', 'stratum'], sort=False)[['pairs', 'covered', 'sum_z2', 'undefined']].sum()
    g['coverage'] = g['covered'] / g['pairs']
    g['z_rms'] = np.sqrt(g['sum_z2'] / g['pairs'])
    return g.drop(columns='sum_z2').reset_index()


def across(full):
    """Readout 2: genes in both datasets detected in both groups of both, LN's LFC and scanpy's in quintiles of LN's
    combined SE; genes absent from a group in either dataset counted apart."""
    a = full[full['dataset'] == 'PBMC3k'].set_index(['type', 'gene'])
    b = full[full['dataset'] == 'PBMC10k'].set_index(['type', 'gene'])
    j = a.join(b, how='inner', lsuffix='_3k', rsuffix='_10k')
    present = ((j['detecting_own_3k'] > 0) & (j['detecting_rest_3k'] > 0) & (j['detecting_own_10k'] > 0)
               & (j['detecting_rest_10k'] > 0))
    absent = int((~present & ((j['detecting_own_3k'] + j['detecting_rest_3k'] > 0)
                              | (j['detecting_own_10k'] + j['detecting_rest_10k'] > 0))).sum())
    j = j[present].copy()
    j['se_combined'] = np.sqrt(j['ln_se_3k'] ** 2 + j['ln_se_10k'] ** 2)
    j['bin'] = pd.qcut(j['se_combined'], BINS, labels=False)
    j['ln_diff'] = (j['ln_lfc_10k'] - j['ln_lfc_3k']).abs()
    j['scanpy_diff'] = (j['scanpy_lfc_10k'] - j['scanpy_lfc_3k']).abs()
    rows = []
    for k, g in j.groupby('bin'):
        rows.append({'bin': int(k) + 1, 'se_low': g['se_combined'].min(), 'se_high': g['se_combined'].max(),
                     'pairs': len(g), 'ln_median_abs_diff': g['ln_diff'].median(),
                     'ln_pearson': pearsonr(g['ln_lfc_3k'], g['ln_lfc_10k']).statistic,
                     'ln_spearman': spearmanr(g['ln_lfc_3k'], g['ln_lfc_10k']).statistic,
                     'ln_covered': float((g['ln_diff'] / g['se_combined'] < Z).mean()),
                     'scanpy_median_abs_diff': g['scanpy_diff'].median(),
                     'scanpy_pearson': pearsonr(g['scanpy_lfc_3k'], g['scanpy_lfc_10k']).statistic,
                     'scanpy_spearman': spearmanr(g['scanpy_lfc_3k'], g['scanpy_lfc_10k']).statistic})
    bins = pd.DataFrame(rows)
    overall = pd.DataFrame([{'pairs': len(j), 'absent_from_a_group': absent,
                             'spearman_absdiff_se': spearmanr(j['ln_diff'], j['se_combined']).statistic,
                             'scanpy_spearman_absdiff_se': spearmanr(j['scanpy_diff'], j['se_combined']).statistic}])
    genes = j.reset_index()[['type', 'gene', 'ln_lfc_3k', 'ln_lfc_10k', 'ln_se_3k', 'ln_se_10k', 'se_combined', 'bin',
                             'scanpy_lfc_3k', 'scanpy_lfc_10k']]
    return bins, overall, genes


def decomposition(full):
    """Readout 3: LN's significant up genes with their parts and class; counts and SE shares per class; and the parts'
    correlation between the datasets on the genes significant up in both."""
    up = full[(full['ln_padj'] < 0.05) & (full['ln_lfc'] > 0)].copy()
    up['det_share'] = up['det'] / up['ln_lfc']
    up['class'] = np.select([up['det_share'] > DETECTION, up['det_share'] < LEVEL], ['detection', 'level'], 'mixed')
    se2 = up[['se2_det_own', 'se2_lvl_own', 'se2_det_rest', 'se2_lvl_rest']]
    for c in se2:
        up[c.replace('se2', 'share')] = up[c] / se2.sum(axis=1)
    cols = ['share_det_own', 'share_lvl_own', 'share_det_rest', 'share_lvl_rest']
    summary = up.groupby(['dataset', 'class']).agg(genes=('gene', 'size'), median_lfc=('ln_lfc', 'median'),
                                                   **{c: (c, 'median') for c in cols}).reset_index()
    counts = up.groupby(['dataset', 'type', 'class']).size().unstack('class', fill_value=0).reset_index()
    a = up[up['dataset'] == 'PBMC3k'].set_index(['type', 'gene'])
    b = up[up['dataset'] == 'PBMC10k'].set_index(['type', 'gene'])
    j = a.join(b, how='inner', lsuffix='_3k', rsuffix='_10k')
    corr = pd.DataFrame([{'part': p, 'pairs': len(j), 'pearson': pearsonr(j[f'{p}_3k'], j[f'{p}_10k']).statistic,
                          'spearman': spearmanr(j[f'{p}_3k'], j[f'{p}_10k']).statistic}
                         for p in ('ln_lfc', 'det', 'lvl')])
    keep = ['dataset', 'type', 'gene', 'ln_lfc', 'ln_se', 'det', 'lvl', 'det_share', 'class', 'detecting_own',
            'detecting_rest'] + cols
    return up[keep], summary, counts, corr


CARRY = {'det': 'LN detection part', 'ln_lfc': 'LN LFC', 'lvl': 'LN level part', 'log1p_effect': 'log1p effect',
         'scanpy_lfc': "scanpy's LFC", 'auc': 'Wilcoxon AUC'}


def carryover(full, adatas, types):
    """Each effect between the datasets on the genes all three methods call up in both (Bonferroni, positive score),
    pooled over the types; the AUC against the other included types, as replication.aucs."""
    shared = adatas['PBMC3k'].var_names.intersection(adatas['PBMC10k'].var_names)
    up = lambda d, m: set(d.loc[(d[f'{m}_padj'] < 0.05) & (d[f'{m}_score'] > 0), 'gene'])
    genes = {}
    for t in types:
        sets = [up(full[(full['dataset'] == n) & (full['type'] == t)], m) for n in rep.DATASETS for m in ('ln', 't', 'w')]
        genes[t] = sorted(set.intersection(*sets) & set(shared))
    values = {}
    for n, adata in adatas.items():
        auc = rep.aucs(adata, sorted(set().union(*genes.values())), types)
        d = full[full['dataset'] == n].set_index(['type', 'gene'])
        for t in types:
            v = d.loc[t].reindex(genes[t])
            values[n, t] = v.assign(auc=auc.loc[genes[t], t].to_numpy(dtype=float))
    rows = []
    for m, label in CARRY.items():
        x = np.concatenate([values['PBMC3k', t][m].to_numpy(float) for t in types])
        y = np.concatenate([values['PBMC10k', t][m].to_numpy(float) for t in types])
        rows.append({'measure': m, 'label': label, 'pairs': len(x), 'pearson': pearsonr(x, y).statistic,
                     'spearman': spearmanr(x, y).statistic})
    return pd.DataFrame(rows)


def forest(full):
    """For the example comparison, the genes with the largest and the smallest LN LFC in each detection stratum."""
    e = full[(full['dataset'] == EXAMPLE[0]) & (full['type'] == EXAMPLE[1])
             & (full['detecting_own'] + full['detecting_rest'] > 0)].copy()
    e['stratum'] = stratum(e['sparser'].to_numpy())
    rows = []
    for _, _, label in reversed(STRATA):
        s = e[e['stratum'] == label].sort_values(['ln_lfc', 'gene'])
        if len(s):
            rows += [s.iloc[-1], s.iloc[0]]
    return pd.DataFrame(rows)[['gene', 'stratum', 'ln_lfc', 'ln_se', 'ln_padj', 'scanpy_lfc', 'detecting_own',
                               'detecting_rest']]


PARTS = [('CD14 Mono', 'LYZ'), ('CD14 Mono', 'FCN1'), ('CD14 Mono', 'CD14'), ('CD14 Mono', 'FTL'), ('B', 'MS4A1'),
         ('B', 'CD79A')]


def part_examples(full):
    rows = []
    for t, g in PARTS:
        for n in rep.DATASETS:
            r = full[(full['dataset'] == n) & (full['type'] == t) & (full['gene'] == g)].iloc[0]
            rows.append({'type': t, 'gene': g, 'dataset': n, 'det': r['det'], 'lvl': r['lvl'], 'ln_lfc': r['ln_lfc']})
    return pd.DataFrame(rows)


def marker_table(full, name, t, rule, universe):
    """One type's markers in one dataset under rule, best first: 'detection', the genes whose detection part is
    significantly positive on its own test (the detection score above the Bonferroni quantile over the genes tested),
    ranked by the detection score; 'detection_ln', the same ranking among LN's significant up genes; or a method's
    practice rule (Bonferroni-significant, up, ranked by score, as replication.markers). Within universe; ties by LN's
    score, then name."""
    d = full[(full['dataset'] == name) & (full['type'] == t)]
    if rule in ('ln', 't_test', 'wilcoxon'):
        p = {'ln': 'ln', 't_test': 't', 'wilcoxon': 'w'}[rule]
        res = pd.DataFrame({'names': d['gene'].to_numpy(), 'scores': d[f'{p}_score'].to_numpy(),
                            'pvals_adj': d[f'{p}_padj'].to_numpy(), 'logfoldchanges': d['ln_lfc'].to_numpy()})
        return rep.markers(res, universe, 'score')['names'].tolist()
    z = d['det'] / np.sqrt(d['se2_det_own'] + d['se2_det_rest'])
    keep = (z > norm.isf(0.025 / len(d))) if rule == 'detection' else (d['ln_padj'] < 0.05) & (d['ln_score'] > 0)
    e = d[keep & d['gene'].isin(universe)].assign(det_z=z)
    return e.sort_values(['det_z', 'ln_score', 'gene'], ascending=[False, False, True])['gene'].tolist()


MARKER_RULES = {'detection': 'LN detection score', 'ln': f"{method_label('ln')} (score rank)", 't_test': method_label('t_test'),
                'wilcoxon': method_label('wilcoxon'), 'detection_ln': method_label('ln')}   # the adopted rule is LN's default
MARKER_NS = (10, 20, 50)


def marker_concordance(full, types):
    """Each rule's top N in PBMC3k found again in PBMC10k's top N, pooled over the types and per type; the practice
    rules must reproduce replication/concordance_pooled.csv."""
    universe = {'PBMC3k': rep.annotation('PBMC10k'), 'PBMC10k': rep.annotation('PBMC3k')}
    tops = {(n, t, r): marker_table(full, n, t, r, universe[n]) for n in rep.DATASETS for t in types for r in MARKER_RULES}
    pooled, per_type = [], []
    for r in MARKER_RULES:
        for n in MARKER_NS:
            both = union = n3 = 0
            for t in types:
                a, b = set(tops['PBMC3k', t, r][:n]), set(tops['PBMC10k', t, r][:n])
                both += len(a & b); union += len(a | b); n3 += len(a)
                per_type.append({'rule': r, 'n': n, 'type': t, 'found_again': len(a & b) / max(len(a), 1)})
            pooled.append({'rule': r, 'label': MARKER_RULES[r], 'n': n, 'found_again': both / n3, 'jaccard': both / union})
    pooled = pd.DataFrame(pooled)
    ref = pd.read_csv(require_input(rep.OUT / 'concordance_pooled.csv', what='the replication concordance',
                                    source='python -m clustering.replication')).set_index(['ranking', 'method', 'n'])
    for r in ('ln', 't_test', 'wilcoxon'):
        for n in MARKER_NS:
            got = pooled.set_index(['rule', 'n']).loc[(r, n), 'found_again']
            if not np.isclose(got, ref.loc[('score', r, n), 'share_found_again']):
                raise SystemExit(f'{r} at N = {n} does not reproduce concordance_pooled.csv; nothing written')
    return pooled, pd.DataFrame(per_type), tops


LIST_TYPES = ('Naive CD4 T', 'CD14 Mono', 'B')
LIST_N = 10


def lists(tops, name='PBMC3k'):
    """LN's top markers in one dataset by its score and by its marker rule (its calls by the detection score)."""
    return pd.DataFrame([{'type': t, 'rule': label, 'rank': k + 1, 'gene': g}
                         for t in LIST_TYPES for r, label in (('ln', 'score'), ('detection_ln', 'detection score'))
                         for k, g in enumerate(tops[name, t, r][:LIST_N])])


def run():
    full, per, hist, adatas = [], [], [], {}
    types = None
    for name in rep.DATASETS:
        adata = rep.load(name)
        t = sorted(adata.obs['type'].cat.categories)
        assert types is None or t == types, 'the datasets have different types'
        types = t
        f = full_data(name, adata, types)
        p, h = split_halves(name, adata, types, f)
        full.append(f), per.append(p), hist.append(h)
        adatas[name] = adata
    full = pd.concat(full, ignore_index=True)
    per = pd.concat(per, ignore_index=True)
    mc, mc_type, tops = marker_concordance(full, types)
    bins, overall, genes = across(full)
    up, summary, counts, corr = decomposition(full)
    example = full[(full['dataset'] == EXAMPLE[0]) & (full['type'] == EXAMPLE[1])
                   & (full['detecting_own'] + full['detecting_rest'] > 0)]
    example = example[['gene', 'ln_lfc', 'ln_se', 'ln_padj', 'scanpy_lfc', 'sparser', 'detecting_own', 'detecting_rest']]
    return {'calibration': calibration(per), 'calibration_by_type': per.groupby(
                ['dataset', 'method', 'stratum', 'type'], sort=False)[['pairs', 'covered']].sum().reset_index(),
            'z_hist': pd.concat(hist, ignore_index=True), 'across_bins': bins, 'across_overall': overall,
            'across_genes': genes, 'decomposition_genes': up, 'decomposition_summary': summary,
            'decomposition_counts': counts, 'decomposition_corr': corr, 'example': example,
            'carryover': carryover(full, adatas, types), 'forest': forest(full), 'part_examples': part_examples(full),
            'marker_concordance': mc, 'marker_concordance_by_type': mc_type, 'lists': lists(tops)}


def strata_axis(ax):
    labels = [lab for _, _, lab in STRATA]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace('>=', '≥') for lab in labels], rotation=30, ha='right')
    ax.set_xlabel('Cells detecting the gene,\nsparser group')


def draw_coverage(ax, cal, name):
    """Readout 1: the share of split-half differences within their 95% interval, by stratum, one line per method."""
    labels = [lab for _, _, lab in STRATA]
    for m in ('t_test', 'ln'):
        d = cal[(cal['dataset'] == name) & (cal['method'] == m)].set_index('stratum').reindex(labels)
        ax.plot(range(len(labels)), d['coverage'], marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                lw=plot_style.ERRORBAR['lw'], color=method_color(m), label=EFFECT[m])
    ax.axhline(0.95, color='0.5', lw=0.8, ls='--')
    strata_axis(ax)
    ax.set_ylabel('Split halves within\ntheir 95% interval')
    ax.set_ylim(0, 1.02)


def draw_z(ax, hist, name, label):
    """Readout 1: the split-half z of one stratum, both methods, against N(0, 1)."""
    for m in ('t_test', 'ln'):
        d = hist[(hist['dataset'] == name) & (hist['method'] == m) & (hist['stratum'] == label)]
        w = HIST[1] - HIST[0]
        ax.step(d['bin'] + w / 2, d['count'] / d['count'].sum() / w, where='mid', color=method_color(m), lw=1,
                label=EFFECT[m])
    x = np.linspace(HIST[0], HIST[-1], 400)
    ax.plot(x, norm.pdf(x), color='0.4', lw=0.8, ls='--', label='N(0, 1)')
    ax.set_xlabel('Split-half z')
    ax.set_ylabel('Density')


def draw_across(ax, bins, column, ylabel):
    """Readout 2: a readout per quintile of LN's combined SE, LN's LFC and scanpy's."""
    x = range(len(bins))
    ax.plot(x, bins[f'ln_{column}'], marker='o', ms=3, lw=1.3, color=method_color('ln'), label="LN's LFC")
    ax.plot(x, bins[f'scanpy_{column}'], marker='o', ms=3, lw=1.3, color='0.45', label="scanpy's LFC")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'{lo:.2g}-{hi:.2g}' for lo, hi in zip(bins['se_low'], bins['se_high'])], rotation=30,
                       ha='right')
    ax.set_xlabel("LN's SE, PBMC3k and PBMC10k combined")
    ax.set_ylabel(ylabel)


CLASS_COLOR = {'detection': '#C0392B', 'mixed': '0.6', 'level': '#2E86C1'}
CLASS_LABEL = {'detection': 'mostly detection', 'mixed': 'mixed', 'level': 'mostly level'}


def draw_decomposition(ax, up, name=EXAMPLE[0], t=EXAMPLE[1], label=12):
    """Readout 3: detection part against level part of LN's significant up genes in one comparison."""
    d = up[(up['dataset'] == name) & (up['type'] == t)]
    for c in ('mixed', 'level', 'detection'):
        e = d[d['class'] == c]
        ax.scatter(e['lvl'], e['det'], s=5, lw=0, alpha=0.8, color=CLASS_COLOR[c], label=CLASS_LABEL[c])
    lim = max(d['det'].max(), d['lvl'].max()) * 1.05
    ax.plot([0, lim], [lim, 0], color='0.7', lw=0.6, ls=':')
    for g in d.nlargest(label, 'ln_lfc')['gene'].tolist() + d.nlargest(4, 'lvl')['gene'].tolist():
        r = d[d['gene'] == g].iloc[0]
        ax.annotate(g, (r['lvl'], r['det']), fontsize=plot_style.SMALL - 1, xytext=(2, 1), textcoords='offset points')
    ax.axhline(0, color='0.85', lw=0.6)
    ax.axvline(0, color='0.85', lw=0.6)
    ax.set_xlabel('Level part (log2 mean among detecting cells)')
    ax.set_ylabel('Detection part (log2 detection rate)')


def draw_example(ax, ex):
    """Readout 4: LN's LFC against its SE in one comparison, coloured by the sparser group's detecting cells."""
    order = ex.sort_values('sparser', ascending=False)
    im = ax.scatter(order['ln_lfc'], order['ln_se'], c=np.log10(order['sparser'] + 1), s=3, lw=0, cmap='viridis')
    ax.set_yscale('log')
    ax.set_xlabel("LN's LFC")
    ax.set_ylabel("LN's SE")
    return im


def draw_example_scanpy(ax, ex):
    """Readout 4: scanpy's LFC against LN's for the same genes; beyond |10| marked."""
    far = ex['scanpy_lfc'].abs() > 10
    ax.scatter(ex.loc[~far, 'ln_lfc'], ex.loc[~far, 'scanpy_lfc'], s=3, lw=0, color='0.45')
    ax.scatter(ex.loc[far, 'ln_lfc'], ex.loc[far, 'scanpy_lfc'], s=4, lw=0, color='#C0392B', label='|scanpy LFC| > 10')
    ax.set_xlabel("LN's LFC")
    ax.set_ylabel("scanpy's LFC")

def draw_forest(ax, fr):
    """LN's LFC with its 95% interval and scanpy's LFC for the forest genes, densest stratum at the top; each label
    gives the cells detecting the gene in the type and in the rest."""
    y = np.arange(len(fr))[::-1]
    ax.errorbar(fr['ln_lfc'], y, xerr=Z * fr['ln_se'], fmt='o', ms=2.5, lw=0.9, capsize=1.5, color=method_color('ln'),
                label="LN's LFC, 95% interval")
    ax.scatter(fr['scanpy_lfc'], y, marker='D', s=9, facecolor='white', edgecolor='0.3', lw=0.7, zorder=3,
               label="scanpy's LFC")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} ({int(a)}/{int(b)})" for g, a, b in zip(fr['gene'], fr['detecting_own'],
                                                                         fr['detecting_rest'])])
    for k in range(2, len(fr), 2):
        ax.axhline(len(fr) - k - 0.5, color='0.88', lw=0.6)
    ax.axvline(0, color='0.75', lw=0.6)
    ax.set_ylim(-0.6, len(fr) - 0.4)
    ax.set_xlabel('log2 fold change')


def draw_se_scatter(ax, genes, t=EXAMPLE[1]):
    """LN's LFC in PBMC3k against PBMC10k for one type, the third of genes with the smallest combined SE and the third
    with the largest."""
    d = genes[genes['type'] == t]
    lo, hi = d['se_combined'].quantile([1 / 3, 2 / 3])
    for sel, color, label in ((d['se_combined'] > hi, '#E5A15A', "LN's SE, largest third"),
                              (d['se_combined'] <= lo, method_color('ln'), "LN's SE, smallest third")):
        e = d[sel]
        ax.scatter(e['ln_lfc_3k'], e['ln_lfc_10k'], s=2, lw=0, alpha=0.6, color=color, label=label, rasterized=True)
    lim = [min(d['ln_lfc_3k'].min(), d['ln_lfc_10k'].min()), max(d['ln_lfc_3k'].max(), d['ln_lfc_10k'].max())]
    ax.plot(lim, lim, color='0.4', lw=0.6, ls='--')
    ax.set_xlabel("LN's LFC, PBMC3k")
    ax.set_ylabel("LN's LFC, PBMC10k")


def carry_color(measure):
    return method_color({'auc': 'wilcoxon', 'scanpy_lfc': 't_test', 'log1p_effect': 't_test'}.get(measure, 'ln'))


def draw_carryover(ax, carry):
    """Each effect's correlation between the datasets on the genes all three methods call up in both: Pearson (solid)
    and Spearman (light)."""
    c = carry.set_index('measure').reindex(list(CARRY))
    y = np.arange(len(c))[::-1]
    for off, col, alpha, label in ((0.2, 'pearson', 1.0, 'Pearson'), (-0.2, 'spearman', 0.45, 'Spearman')):
        ax.barh(y + off, c[col], height=0.38, color=[carry_color(m) for m in c.index], alpha=alpha, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([CARRY[m] for m in c.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('Correlation, 3k against 10k')


def draw_parts(ax, ex):
    """Each example gene's LFC split into its detection part and its level part, PBMC3k and PBMC10k side by side."""
    genes = list(dict.fromkeys(zip(ex['type'], ex['gene'])))
    ticks, labels = [], []
    for i, (t, g) in enumerate(genes):
        for j, n in enumerate(rep.DATASETS):
            r = ex[(ex['type'] == t) & (ex['gene'] == g) & (ex['dataset'] == n)].iloc[0]
            x = 2.6 * i + j
            ax.bar(x, r['det'], width=0.85, color=method_color('ln'), label='Detection part' if i == j == 0 else None)
            bottom = r['det'] if np.sign(r['det']) == np.sign(r['lvl']) else 0
            ax.bar(x, r['lvl'], width=0.85, bottom=bottom, color=method_color('ln'), alpha=0.35,
                   label='Level part' if i == j == 0 else None)
            ticks.append(x)
            labels.append(n.replace('PBMC', ''))
        ax.text(2.6 * i + 0.5, -0.2, g, transform=ax.get_xaxis_transform(), ha='center', va='top')
        if i and t != genes[i - 1][0]:
            ax.axvline(2.6 * i - 0.8, color='0.6', lw=0.6, ls='--')
    for t in dict.fromkeys(t for t, _ in genes):
        xs = [2.6 * i + 0.5 for i, (u, _) in enumerate(genes) if u == t]
        ax.text(sum(xs) / len(xs), -0.34, t, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=plot_style.SMALL, color='0.35')
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=plot_style.SMALL)
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_ylabel('log2 fold change')


def draw_lists(ax, ls):
    """LN's top markers per type, by its score and by its detection score, as a table."""
    ax.axis('off')
    types = list(dict.fromkeys(ls['type']))
    rules = list(dict.fromkeys(ls['rule']))
    n = ls['rank'].max()
    width = 1 / (len(types) * len(rules))
    for i, t in enumerate(types):
        ax.text((i * len(rules) + len(rules) / 2) * width, 1.0, t, ha='center', va='top', fontweight='bold',
                fontsize=plot_style.SMALL + 1, transform=ax.transAxes)
        for j, r in enumerate(rules):
            x = (i * len(rules) + j + 0.5) * width
            ax.text(x, 0.9, r, ha='center', va='top', fontsize=plot_style.SMALL, style='italic',
                    transform=ax.transAxes)
            g = ls[(ls['type'] == t) & (ls['rule'] == r)].sort_values('rank')['gene']
            for k, gene in enumerate(g):
                ax.text(x, 0.8 - k * 0.8 / n, gene, ha='center', va='top', fontsize=plot_style.SMALL,
                        transform=ax.transAxes)


MARKER_STYLE = {'detection_ln': (method_color('ln'), 1.0), 'ln': (method_color('ln'), 0.4),
                't_test': (method_color('t_test'), 1.0), 'wilcoxon': (method_color('wilcoxon'), 1.0)}


def draw_marker_concordance(ax, mc):
    """Each rule's top N in PBMC3k found again in PBMC10k's top N, LN's two rules in its colour, the baselines in
    theirs."""
    rules = list(MARKER_STYLE)
    w = 0.8 / len(rules)
    for i, r in enumerate(rules):
        d = mc[mc['rule'] == r].sort_values('n')
        color, alpha = MARKER_STYLE[r]
        ax.bar(np.arange(len(d)) + (i - (len(rules) - 1) / 2) * w, d['found_again'], width=w, color=color, alpha=alpha,
               label=MARKER_RULES[r])
    ax.set_xticks(range(len(MARKER_NS)))
    ax.set_xticklabels([f'Top {n}' for n in MARKER_NS])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Found again,\nPBMC3k to PBMC10k')


def plot(t):
    plot_style.use()
    for name in rep.DATASETS:
        tag = name.lower()
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_coverage(ax, t['calibration'], name)
        ax.set_title(name)
        rep.save(fig, OUT / fig_name(f'{tag}_coverage'))
        for label in ('1-4', '>=500'):
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_z(ax, t['z_hist'], name, label)
            ax.set_title(f"{name}, {label.replace('>=', '≥')} cells")
            rep.save(fig, OUT / fig_name(f"{tag}_z_{label.replace('>=', 'ge').replace('-', '_')}"))
    for column, ylabel in (('median_abs_diff', '|LFC PBMC10k − LFC PBMC3k|,\nmedian'),
                           ('spearman', 'LFC PBMC3k against PBMC10k,\nSpearman')):
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_across(ax, t['across_bins'], column, ylabel)
        rep.save(fig, OUT / fig_name(f'across_{column}'))
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_decomposition(ax, t['decomposition_genes'])
    ax.set_title(f'{EXAMPLE[1]}, {EXAMPLE[0]}')
    rep.save(fig, OUT / fig_name('decomposition'))
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    im = draw_example(ax, t['example'])
    fig.colorbar(im, ax=ax, label='log10(cells detecting + 1),\nsparser group')
    ax.set_title(f'{EXAMPLE[1]}, {EXAMPLE[0]}')
    rep.save(fig, OUT / fig_name('example_se'), key=False)
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_example_scanpy(ax, t['example'])
    ax.set_title(f'{EXAMPLE[1]}, {EXAMPLE[0]}')
    rep.save(fig, OUT / fig_name('example_scanpy'))
    for stem, draw, size, key in (('forest', lambda ax: draw_forest(ax, t['forest']), (3.2, 2.6), True),
                                  ('se_scatter', lambda ax: draw_se_scatter(ax, t['across_genes']), plot_style.PANEL, True),
                                  ('carryover', lambda ax: draw_carryover(ax, t['carryover']), (3.2, 2.2), True),
                                  ('parts', lambda ax: draw_parts(ax, t['part_examples']), (3.4, 2.2), True),
                                  ('lists', lambda ax: draw_lists(ax, t['lists']), (4.0, 2.2), False),
                                  ('marker_concordance', lambda ax: draw_marker_concordance(ax, t['marker_concordance']),
                                   plot_style.PANEL, True)):
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        rep.save(fig, OUT / fig_name(stem), key=key)


NAMES = ('calibration', 'calibration_by_type', 'z_hist', 'across_bins', 'across_overall', 'across_genes',
         'decomposition_genes', 'decomposition_summary', 'decomposition_counts', 'decomposition_corr', 'example',
         'carryover', 'forest', 'part_examples', 'marker_concordance', 'marker_concordance_by_type', 'lists')


def main():
    parser = argparse.ArgumentParser(description="LN's SE on real data: split halves, across datasets, decomposition")
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in NAMES}
    if args.plot_only:
        t = {k: pd.read_csv(require_input(p, what=f'the lfc_se {k}', source='this script without --plot-only'))
             for k, p in csv.items()}
    else:
        t = run()
        for k, v in t.items():
            v.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(t)


if __name__ == '__main__':
    main()
