#!/usr/bin/env python3
"""LN's marker rule against LN's score and the baselines: are the markers cell-type specific, free of housekeeping
genes, and confirmed by surface protein?

The rule (vault: decision-ln-markers-detection-part): LN's Bonferroni-significant up calls ranked by the detection score,
the detection part log2(p1/p0) over its SE (LN's detection terms), ties by LN's score then name. Against it, the practice
rule (Bonferroni-significant, up, ranked by score) for LN, the log1p t-test and the Wilcoxon. Two settings: the 6 types
of clustering/replication.py against the other included cells, both datasets; and every Leiden cluster at the five
resolutions against all other cells (clustering/de.py's outputs), judged by its majority CellTypist type. Per list:
- the share of the top TOP that are Monaco et al. 2019's immune housekeeping genes (their Table S3: mean > 5 and SD < 1
  across 29 sorted immune populations), and that are ribosomal or mitochondrial;
- the share HPA (release 25.1) calls immune cell enriched, or group enriched with a group holding the type, for the
  cluster's type (HPA_TYPES); with "enhanced" counted, alongside;
- the share confirmed by sorted-cell bulk, multires_gaps.py's truth (the type at least MIN_TPM and BULK_FOLD times the
  rest's mix);
- PBMC10k only: of the genes of the 13 single-gene antibodies (ANTIBODIES) whose protein separates the cluster from the
  rest (per-cell CLR, AUC >= PROTEIN_AUC), the share in the top TOP and the top TOP_PROTEIN, and the share of the mapped
  genes in the top TOP_PROTEIN that are such markers.
Genes are matched by Ensembl ID. Means per setting and resolution; at the fine resolutions (FINE clusters or more) the
rule against each other, paired over clusters with a bootstrap; top-TOP stability between consecutive resolutions; and,
at PBMC10k's fine resolutions, each protein's recovery. The score rules' multi-resolution lists, bulk shares and
stabilities must reproduce multires_gaps.py's.

    python -m fetch_data --arm clustering            # Monaco's supplementary and HPA 25.1, with the arm's other inputs
    python -m clustering.marker_rule                 # from reproducibility/; a few minutes
    python -m clustering.marker_rule --plot-only     # redraw from the CSVs
"""

import argparse
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata

import plot_style
from clustering import lfc_se as L
from clustering import lower_bound as lb
from clustering import multires as mr
from clustering import multires_gaps as mg
from clustering import replication as rep
from clustering.celltype_labels import RAW
from evaluation_utils import _get_rank_genes_df
from paths import fig_name, require_input, results_dir

OUT = results_dir(__file__) / 'marker_rule'
MONACO_SUPP = mg.MONACO.parent / 'monaco2019_supplementary.zip'
HPA = mg.MONACO.parent / 'hpa_25.1_proteinatlas.tsv.zip'
TOP, TOP_PROTEIN = 20, 50
FINE = 18
PROTEIN_AUC = 0.75
BOOT, SEED = 2000, 0
RULES = tuple(L.MARKER_STYLE)  # LN's marker rule ('detection_ln'), LN's score, the log1p t-test, the Wilcoxon
HPA_TYPES = {'Naive CD4 T': ['naive CD4 T-cell'], 'Memory CD4 T': ['memory CD4 T-cell'],
             'CD8 T': ['naive CD8 T-cell', 'memory CD8 T-cell'], 'NK': ['NK-cell'], 'B': ['naive B-cell', 'memory B-cell'],
             'CD14 Mono': ['classical monocyte'], 'FCGR3A Mono': ['non-classical monocyte'], 'DC': ['myeloid DC']}
ANTIBODIES = {'CD3_TotalC': 'CD3E', 'CD19_TotalC': 'CD19', 'CD4_TotalC': 'CD4', 'CD8a_TotalC': 'CD8A',
              'CD14_TotalC': 'CD14', 'CD16_TotalC': 'FCGR3A', 'CD56_TotalC': 'NCAM1', 'CD25_TotalC': 'IL2RA',
              'PD-1_TotalC': 'PDCD1', 'TIGIT_TotalC': 'TIGIT', 'CD127_TotalC': 'IL7R', 'CD197_TotalC': 'CCR7',
              'HLA-DR_TotalC': 'HLA-DRA'}
METRICS = ('housekeeping', 'technical', 'hpa', 'hpa_enhanced', 'bulk', 'protein_recall', 'protein_recall50',
           'protein_precision50')


def housekeeping():
    """Monaco et al. 2019's immune housekeeping genes by Ensembl ID, and every gene of their table."""
    path = require_input(MONACO_SUPP, what="Monaco et al. 2019's supplementary files", source='python -m fetch_data --arm clustering')
    with zipfile.ZipFile(path) as z:
        h = pd.read_excel(io.BytesIO(z.read('mmc4.xlsx')), sheet_name='Immune-housekeeping genes', header=3)
    ens = h['Ensemble ID'].astype(str).str.split('.').str[0]
    return set(ens[(h['mean expression'] > 5) & (h['SD expression'] < 1)]), set(ens)


def hpa_markers():
    """Per type, the Ensembl IDs HPA calls enriched or group enriched for one of its HPA cell types (strict) and with
    'enhanced' counted (loose); and every Ensembl ID of HPA's table."""
    h = pd.read_csv(require_input(HPA, what='HPA 25.1', source='python -m fetch_data --arm clustering'), sep='\t',
                    compression='zip', usecols=['Ensembl', 'RNA blood cell specificity', 'RNA blood cell specific nTPM'])
    cells = h['RNA blood cell specific nTPM'].fillna('').map(lambda v: {c.split(':')[0].strip() for c in v.split(';') if c})
    out = {}
    for key, classes in (('strict', ['Immune cell enriched', 'Group enriched']),
                         ('loose', ['Immune cell enriched', 'Group enriched', 'Immune cell enhanced'])):
        cls = h['RNA blood cell specificity'].isin(classes)
        out[key] = {t: set(h.loc[cls & cells.map(lambda s, c=set(c): bool(s & c)), 'Ensembl']) for t, c in HPA_TYPES.items()}
    return out, set(h['Ensembl'])


def proteins():
    """PBMC10k's antibodies, per-cell CLR over all of them."""
    full = sc.read_10x_h5(str(require_input(RAW, what="the 5' PBMC 10k matrix", source='python -m fetch_data --arm citeseq')),
                          gex_only=False)
    full.var_names_make_unique()
    ab = full.var_names[full.var['feature_types'] == 'Antibody Capture']
    x = np.log1p(pd.DataFrame(full[:, ab].X.toarray(), index=full.obs_names, columns=ab))
    return x.sub(x.mean(1), axis=0)


def auc(x, own):
    r = rankdata(x)
    n1, n2 = own.sum(), (~own).sum()
    return (r[own].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def protein_markers(clr, cells, own, genes):
    c = clr.loc[cells]
    return {g for p, g in ANTIBODIES.items() if g in genes and auc(c[p].to_numpy(), own) >= PROTEIN_AUC}


class Judge:
    """One dataset's truths: Ensembl IDs of its genes, the sorted-bulk profiles, and the references."""

    def __init__(self, name, genes, hk, hpa, clr):
        self.name, self.hk, self.hpa, self.clr = name, hk, hpa, clr
        e = mg.ensembl_ids(name, pd.Index(genes))
        self.ens = dict(zip(genes, e))
        profiles = mg.bulk_profiles()
        matched = e.notna() & e.isin(profiles.index)
        self.bulk = profiles.reindex(e[matched].to_numpy())
        self.bulk.index = pd.Index(genes)[matched.to_numpy()]

    def bulk_true(self, ref, rest_types):
        if ref not in mg.BULK:
            return None
        rest = pd.Series(rest_types).value_counts()
        rest = rest[rest.index.isin(list(mg.BULK))]
        rp = (self.bulk[rest.index] * rest.to_numpy()).sum(axis=1) / rest.sum()
        t = self.bulk[ref]
        return set(self.bulk.index[(t >= mg.MIN_TPM) & (t >= mg.BULK_FOLD * rp)])

    def score(self, top, ref, true, cells=None, own=None):
        out = dict.fromkeys(METRICS, np.nan)
        head = top[:TOP]
        known = [e for e in (self.ens.get(g) for g in head) if isinstance(e, str)]
        in_hk = [e for e in known if e in self.hk[1]]
        if in_hk:
            out['housekeeping'] = np.mean([e in self.hk[0] for e in in_hk])
        out['technical'] = lb.technical_share(head)
        in_hpa = [e for e in known if e in self.hpa[1]]
        if ref in HPA_TYPES and in_hpa:
            out['hpa'] = np.mean([e in self.hpa[0]['strict'][ref] for e in in_hpa])
            out['hpa_enhanced'] = np.mean([e in self.hpa[0]['loose'][ref] for e in in_hpa])
        judge = [g for g in head if g in self.bulk.index]
        if true is not None and judge:
            out['bulk'] = np.mean([g in true for g in judge])
        if self.clr is not None and own is not None:
            pm = protein_markers(self.clr, cells, own, set(self.ens))
            if pm:
                out['protein_recall'] = len(set(top[:TOP]) & pm) / len(pm)
                out['protein_recall50'] = len(set(top[:TOP_PROTEIN]) & pm) / len(pm)
            mapped = [g for g in top[:TOP_PROTEIN] if g in set(ANTIBODIES.values())]
            if mapped:
                out['protein_precision50'] = np.mean([g in pm for g in mapped])
            out['protein_markers'] = ';'.join(sorted(pm))
        return out


def detection_ln(adata, det, lab, c, key):
    """The rule's ordering for one multi-resolution comparison."""
    own = lab == c
    n1, n0 = own.sum(), (~own).sum()
    a1 = np.maximum(np.asarray(det[own].sum(0)).ravel().astype(float), 1)
    a0 = np.maximum(np.asarray(det[~own].sum(0)).ravel().astype(float), 1)
    z = np.log((a1 / n1) / (a0 / n0)) / np.sqrt(1 / a1 - 1 / n1 + 1 / a0 - 1 / n0)
    ln = _get_rank_genes_df(adata, c, f'ln_{key}').set_index('names').reindex(adata.var_names)
    score = ln['scores'].astype(float).to_numpy()
    d = pd.DataFrame({'gene': adata.var_names.astype(str), 'z': z, 'ln_score': score})
    d = d[(ln['pvals_adj'] < 0.05).to_numpy() & (score > 0)]
    return d.sort_values(['z', 'ln_score', 'gene'], ascending=[False, False, True])['gene'].tolist()


def run():
    hk, hpa = housekeeping(), hpa_markers()
    clr = proteins()
    per_method = pd.read_csv(require_input(mg.OUT / 'per_method.csv', what='the multires_gaps per-method table',
                                           source='python -m clustering.multires_gaps'), dtype={'cluster': str})
    rows, lists = [], []
    for name in rep.DATASETS:
        # the 6 types
        a = rep.load(name)
        types = sorted(a.obs['type'].cat.categories)
        full = L.full_data(name, a, types)
        judge = Judge(name, a.var_names.astype(str), hk, hpa, clr if name == 'PBMC10k' else None)
        labels = a.obs['type'].astype(str).to_numpy()
        universe = set(a.var_names.astype(str))
        for t in types:
            own = labels == t
            true = judge.bulk_true(t, labels[~own])
            for r in RULES:
                top = L.marker_table(full, name, t, r, universe)[:TOP_PROTEIN]
                rows.append({'dataset': name, 'setting': 'types', 'resolution': 'types', 'clusters': len(types),
                             'cluster': t, 'reference': t, 'rule': r, **judge.score(top, t, true, a.obs_names, own)})
                lists.append({'dataset': name, 'resolution': 'types', 'cluster': t, 'rule': r, 'top': ';'.join(top)})
        # every cluster at the five resolutions
        adata, cp, coarse, keys, _ = mr.load(name)
        judge = Judge(name, adata.var_names.astype(str), hk, hpa, clr if name == 'PBMC10k' else None)
        det = (cp > 0).tocsr()
        ctypes = coarse.fillna('(unmapped)').to_numpy()
        for key in keys:
            lab = adata.obs[key].astype(str).to_numpy()
            ids = sorted(np.unique(lab), key=lambda v: int(v) if v.isdigit() else v)
            for c in ids:
                own = lab == c
                counts = pd.Series(ctypes[own]).value_counts().rename_axis('type').reset_index()
                ref = counts.sort_values(['count', 'type'], ascending=[False, True])['type'].iloc[0]
                ref = None if ref == '(unmapped)' else ref
                true = judge.bulk_true(ref, ctypes[~own])
                tops = {RULES[0]: detection_ln(adata, det, lab, c, key)[:TOP_PROTEIN]}
                for m in RULES[1:]:
                    tops[m] = mr.ranked(adata, c, f'{m}_{key}')[:TOP_PROTEIN]
                    stored = per_method[(per_method.dataset == name) & (per_method.resolution == key)
                                        & (per_method.cluster == c) & (per_method.method == m)]['top'].iloc[0]
                    if ';'.join(tops[m][:TOP]) != (stored if isinstance(stored, str) else ''):
                        raise SystemExit(f'{name} {key} {c} {m}: the top {TOP} does not reproduce per_method.csv')
                for m, top in tops.items():
                    rows.append({'dataset': name, 'setting': 'resolutions', 'resolution': key, 'clusters': len(ids),
                                 'cluster': c, 'reference': ref, 'rule': m,
                                 **judge.score(top, ref, true, adata.obs_names, own)})
                    lists.append({'dataset': name, 'resolution': key, 'cluster': c, 'rule': m, 'top': ';'.join(top)})
        print(f'  {name} done', flush=True)
    per = pd.DataFrame(rows)
    check = per[(per.setting == 'resolutions') & per.rule.isin(RULES[1:])].merge(
        per_method, left_on=['dataset', 'resolution', 'cluster', 'rule'], right_on=['dataset', 'resolution', 'cluster', 'method'])
    if not np.allclose(check['bulk'], check['bulk_precision'], equal_nan=True):
        raise SystemExit('the score rules do not reproduce multires_gaps per_method.csv bulk_precision; nothing written')
    return per, pd.DataFrame(lists)


def summarise(per):
    g = per.groupby(['dataset', 'setting', 'resolution', 'clusters', 'rule'], sort=False)
    return g[list(METRICS)].mean().reset_index()


def paired(per):
    """At the fine resolutions, per dataset and metric, the rule minus each other: mean and 95% bootstrap interval."""
    rng = np.random.default_rng(SEED)
    rows = []
    fine = per[(per.setting == 'resolutions') & (per.clusters >= FINE)]
    for name, d in fine.groupby('dataset', sort=False):
        for metric in METRICS:
            w = d.pivot_table(index=['resolution', 'cluster'], columns='rule', values=metric)
            for other in RULES[1:]:
                if RULES[0] not in w or other not in w:
                    continue
                diff = (w[RULES[0]] - w[other]).dropna().to_numpy()
                if len(diff) < 5:
                    continue
                boot = [rng.choice(diff, len(diff)).mean() for _ in range(BOOT)]
                rows.append({'dataset': name, 'metric': metric, 'versus': other, 'comparisons': len(diff),
                             'mean_difference': diff.mean(), 'low': np.percentile(boot, 2.5), 'high': np.percentile(boot, 97.5),
                             'better': int((diff > 0).sum()), 'equal': int((diff == 0).sum()), 'worse': int((diff < 0).sum())})
    return pd.DataFrame(rows)


def stability(lists):
    """Top-TOP Jaccard of a cell's cluster at consecutive resolutions, mean over cells; the score rules must reproduce
    multires_gaps.py's."""
    rows = []
    ref = pd.read_csv(require_input(mg.OUT / 'stability.csv', what='the multires_gaps stability',
                                    source='python -m clustering.multires_gaps'))
    for name in rep.DATASETS:
        adata, _, _, keys, _ = mr.load(name)
        order = sorted(keys, key=lambda k: adata.obs[k].nunique())
        tops = {(r.resolution, r.cluster, r.rule): set(str(r.top).split(';')[:TOP]) - {'nan', ''}
                for r in lists[lists.dataset == name].itertuples()}
        for k1, k2 in zip(order[:-1], order[1:]):
            pairs = pd.Series(list(zip(adata.obs[k1].astype(str), adata.obs[k2].astype(str)))).value_counts()
            for m in RULES:
                jac = sum(n * len(tops[k1, a, m] & tops[k2, b, m]) / max(len(tops[k1, a, m] | tops[k2, b, m]), 1)
                          for (a, b), n in pairs.items()) / adata.n_obs
                rows.append({'dataset': name, 'from': k1, 'to': k2, 'clusters_from': adata.obs[k1].nunique(),
                             'clusters_to': adata.obs[k2].nunique(), 'rule': m, 'top_jaccard': jac})
    st = pd.DataFrame(rows)
    chk = st[st.rule != RULES[0]].merge(ref, left_on=['dataset', 'from', 'to', 'rule'], right_on=['dataset', 'from', 'to', 'method'])
    if len(chk) != (st.rule != RULES[0]).sum() or not np.allclose(chk['top_jaccard_x'], chk['top_jaccard_y']):
        raise SystemExit('the score rules do not reproduce multires_gaps stability.csv; nothing written')
    return st


def protein_genes(per, lists):
    """PBMC10k's fine resolutions: per protein, the share of the comparisons where it is a protein marker whose top TOP
    holds its gene, per rule; and how often each rule lists the gene where the protein is not a marker."""
    fine = per[(per.dataset == 'PBMC10k') & (per.setting == 'resolutions') & (per.clusters >= FINE)]
    tops = {(r.resolution, r.cluster, r.rule): str(r.top).split(';')[:TOP] for r in lists[lists.dataset == 'PBMC10k'].itertuples()}
    rows = []
    for g in ANTIBODIES.values():
        row = {'gene': g}
        for m in RULES:
            d = fine[fine.rule == m]
            hit = [g in tops[r.resolution, r.cluster, m] for r in d.itertuples()]
            is_marker = d['protein_markers'].fillna('').str.split(';').apply(lambda v: g in v).to_numpy()
            row['comparisons'] = int(is_marker.sum())
            row[m] = np.mean(np.array(hit)[is_marker]) if is_marker.any() else np.nan
            row[f'{m}_listed_not_marker'] = int((np.array(hit) & ~is_marker).sum())
        rows.append(row)
    return pd.DataFrame(rows)


LABELS = {'housekeeping': 'Housekeeping genes\nin top 20', 'technical': 'Ribosomal and mito-\nchondrial genes in top 20',
          'hpa': 'HPA-enriched for the\ntype, fraction of top 20', 'hpa_enhanced': 'HPA enriched or enhanced,\nfraction of top 20',
          'bulk': 'Confirmed by sorted\nbulk, fraction of top 20', 'protein_recall': 'Protein markers\nin top 20 (recall)',
          'protein_recall50': 'Protein markers\nin top 50 (recall)', 'protein_precision50': 'Mapped genes in top 50\nthat are protein markers'}


def style(m):
    """lfc_se's marker colours: LN's two rules in its colour, the baselines in theirs."""
    color, alpha = L.MARKER_STYLE[m]
    return dict(color=color, alpha=alpha, label=L.MARKER_RULES[m])


TITLES = {'housekeeping': 'Housekeeping', 'protein_recall': 'Protein markers'}   # for plots side by side


def draw_by_resolution(ax, per, name, metric, rules=RULES, ylabel=None):
    """A metric's mean over each resolution's clusters with bars of one standard error, one colour per rule, as Fig 1's
    error bars, the resolutions evenly spaced; one y scale for both datasets."""
    d = per[(per.dataset == name) & (per.setting == 'resolutions')]
    g = per[per.setting == 'resolutions'].groupby(['dataset', 'clusters', 'rule'])[metric]
    top = (g.mean() + g.sem().fillna(0)).max()
    sizes = sorted(d['clusters'].unique())   # evenly spaced, as multires_gaps.draw_gaps
    for m in reversed(rules):
        e = d[d.rule == m].groupby('clusters')[metric].agg(['mean', 'sem'])
        color, alpha = L.MARKER_STYLE[m]
        ax.errorbar([sizes.index(n) for n in e.index], e['mean'], yerr=e['sem'], color=color, alpha=alpha,
                    label=L.MARKER_RULES[m], clip_on=False, **plot_style.ERRORBAR)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(LABELS[metric] if ylabel is None else ylabel)
    ax.set_ylim(0, 1.08 * top)


def draw_protein_genes(ax, genes, rules=RULES):
    """Each protein's recovery in the top TOP at PBMC10k's fine resolutions, one bar per rule."""
    g = genes[genes['comparisons'] > 0].sort_values(['comparisons', 'gene'], ascending=[True, False]).reset_index(drop=True)
    w = 0.8 / len(rules)
    for i, m in enumerate(rules):
        ax.barh(np.arange(len(g)) + (len(rules) / 2 - i - 0.5) * w, g[m], height=w, **style(m))
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{x} ({n})" for x, n in zip(g['gene'], g['comparisons'])])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel(f'Fraction recovered in top {TOP}')


def plot(per, genes):
    plot_style.use()
    for name in rep.DATASETS:
        for metric in METRICS:
            if metric.startswith('protein') and name != 'PBMC10k':
                continue
            fig, ax = plt.subplots(figsize=plot_style.PANEL)
            draw_by_resolution(ax, per, name, metric)
            ax.set_title(name)
            rep.save(fig, OUT / fig_name(f'{name.lower()}_{metric}'))
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    draw_protein_genes(ax, genes)
    ax.set_title(f'PBMC10k, at least {FINE} clusters')
    rep.save(fig, OUT / fig_name('protein_genes'))


NAMES = ('per_list', 'lists', 'summary', 'paired', 'stability', 'protein_genes')


def main():
    parser = argparse.ArgumentParser(description="LN's marker rule against housekeeping genes, HPA, sorted bulk and protein")
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in NAMES}
    if args.plot_only:
        t = {k: pd.read_csv(require_input(p, what=f'the marker_rule {k}', source='this script without --plot-only'),
                            dtype={'cluster': str}) for k, p in csv.items()}
    else:
        per, lists = run()
        t = {'per_list': per, 'lists': lists, 'summary': summarise(per), 'paired': paired(per),
             'stability': stability(lists), 'protein_genes': protein_genes(per, lists)}
        for k, v in t.items():
            v.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
        print(t['paired'].round(3).to_string(index=False))
    plot(t['per_list'], t['protein_genes'])


if __name__ == '__main__':
    main()
