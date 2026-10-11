#!/usr/bin/env python3
"""Do the methods select the same markers in both of the arm's datasets, PBMC3k and PBMC10k?

The cell types clustering/celltype_labels.py includes are each tested against the others in both datasets, the tests
run as clustering/de.py runs them. A method's markers of a type are its Bonferroni-significant genes with a positive
score, ranked by its score (or, as a sensitivity, by its reported LFC), among the genes the other dataset's annotation
has. Concordance, for the method's top-N lists A (PBMC3k) and B (PBMC10k): |A ∩ B| / |A ∪ B|, the share of all genes
it selected that it selected in both. Effect transfer: per type, over the genes detected in at least DETECTED of the
type's or the reference's cells in each dataset, LN's LFC, scanpy's logfoldchanges and AUC - 1/2 (U / (n1 n2), ties
counted one half) in PBMC3k against PBMC10k, and the same over each dataset's top-TOP_TRANSFER markers.

    python -m clustering.replication               # from reproducibility/, minutes
    python -m clustering.replication --plot-only   # redraw from the CSVs
"""

import argparse
import tarfile

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml
from scipy.stats import rankdata, spearmanr

import plot_style
from clustering.celltype_labels import RAW, dataset_paths
from clustering.heldout_fp import de_output
from evaluation_utils import _get_rank_genes_df
from lntest import rank_genes_groups_ln
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

DATASETS = ('PBMC3k', 'PBMC10k')
METHODS = ('ln', 't_test', 'wilcoxon')
RANKINGS = {'lfc': 'logfoldchanges', 'score': 'scores'}
RANKING = 'score'        # the practice rule's; the other is the sensitivity
NS = (5, 10, 20, 50)
N_MAIN = 10
DETECTED = 0.1
TOP_TRANSFER = 20
# The effect measures, each with the method whose colour it takes
MEASURES = {'ln_lfc': ("LN's LFC", 'ln'), 'scanpy_lfc': ("scanpy's LFC", 't_test'), 'auc': ('AUC − ½', 'wilcoxon')}
CHUNK = 1000             # genes ranked at a time
OUT = results_dir(__file__) / 'replication'


def load(name):
    """A dataset's included cells: CP10K and log1p CP10K over the arm's genes, and each cell's type."""
    config_path, out = dataset_paths()[name]
    path = require_input(de_output(yaml.safe_load(open(config_path))['adata_path'], str(out)),
                         what=f'the {name} DE-annotated AnnData', source='python -m clustering.de (see README)')
    labels = pd.read_csv(require_input(out / 'metrics' / 'celltypes.csv', what=f'the {name} cell-type labels',
                                       source='python -m clustering.celltype_labels'), index_col=0)
    with h5py.File(path, 'r') as f:
        obs, var = ad.io.read_elem(f['obs']), ad.io.read_elem(f['var'])
        layers = {k: sp.csr_matrix(ad.io.read_elem(f[f'layers/{k}'])) for k in ('norm_counts', 'log1p_norm')}
    keep = labels.loc[obs.index, 'included'].to_numpy(dtype=bool)
    cells = obs.index[keep]
    adata = ad.AnnData(X=layers['norm_counts'][keep], var=pd.DataFrame(index=var.index),
                       obs=pd.DataFrame({'type': pd.Categorical(labels.loc[cells, 'coarse'])}, index=cells))
    adata.layers['norm_counts'] = adata.X
    adata.layers['log1p_norm'] = layers['log1p_norm'][keep]
    return adata


def annotation(name):
    """Every gene of a dataset's 10x annotation, named as the arm names them (scanpy's unique symbols)."""
    if name == 'PBMC10k':
        with h5py.File(require_input(RAW, what="the 5' PBMC 10k matrix", source='python -m fetch_data --arm citeseq'),
                       'r') as f:
            names = f['matrix/features/name'][:].astype(str)
            kinds = f['matrix/features/feature_type'][:].astype(str)
        return set(ad.utils.make_index_unique(pd.Index(names))[kinds == 'Gene Expression'])
    config_path, _ = dataset_paths()[name]
    tar = require_input(yaml.safe_load(open(config_path))['adata_path'], what='the PBMC3k 10x archive',
                        source='python -m fetch_data --arm clustering')
    with tarfile.open(tar) as tf:
        genes = pd.read_csv(tf.extractfile(next(m for m in tf.getmembers() if m.name.endswith('genes.tsv'))),
                            sep='\t', header=None)
    return set(ad.utils.make_index_unique(pd.Index(genes[1].astype(str))))


def run_tests(adata, groupby='type'):
    """Each group against the others, as clustering/de.py runs the three tests."""
    rank_genes_groups_ln(adata, groupby, sparse=True, key_added='ln', layer='norm_counts', corr_method='bonferroni')
    sc.tl.rank_genes_groups(adata, groupby=groupby, method='t-test', corr_method='bonferroni', key_added='t_test',
                            layer='log1p_norm', use_raw=False)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon', corr_method='bonferroni', key_added='wilcoxon',
                            layer='log1p_norm', use_raw=False, tie_correct=True)


def markers(res, universe, ranking):
    """A method's markers of one type, best first: Bonferroni-significant with a positive score, among universe,
    ranked by the LFC or the score; ties broken by the score, then the name."""
    d = res[(res['pvals_adj'] < 0.05) & (res['scores'] > 0) & res['names'].isin(universe)]
    keys = ['logfoldchanges', 'scores', 'names'] if ranking == 'lfc' else ['scores', 'names']
    return d.sort_values(keys, ascending=[False] * (len(keys) - 1) + [True])


def marker_table(data, universe, types):
    """Every dataset's, ranking's, method's and type's top max(NS) markers, with their rank; and, per dataset and
    method, the significant upregulated genes set aside for want of a match in the other annotation."""
    rows, skipped = [], []
    for name in DATASETS:
        for m in METHODS:
            up_total, outside = 0, 0
            for t in types:
                res = _get_rank_genes_df(data[name], t, m)
                up = res[(res['pvals_adj'] < 0.05) & (res['scores'] > 0)]
                up_total += len(up)
                outside += int((~up['names'].isin(universe[name])).sum())
                for ranking in RANKINGS:
                    top = markers(res, universe[name], ranking).head(max(NS))
                    rows += [{'dataset': name, 'ranking': ranking, 'method': m, 'type': t, 'rank': k + 1,
                              'gene': r.names, 'logfoldchanges': r.logfoldchanges, 'scores': r.scores,
                              'pvals_adj': r.pvals_adj} for k, r in enumerate(top.itertuples())]
            skipped.append({'dataset': name, 'method': m, 'significant_up': up_total, 'outside_other_annotation': outside})
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def concordance(table):
    """Per ranking, method, type and N, the top-N lists' overlap: |A ∩ B| / |A ∪ B| and |A ∩ B| / |A|."""
    rows = []
    for (ranking, m, t), d in table.groupby(['ranking', 'method', 'type']):
        for n in NS:
            a = set(d[(d['dataset'] == 'PBMC3k') & (d['rank'] <= n)]['gene'])
            b = set(d[(d['dataset'] == 'PBMC10k') & (d['rank'] <= n)]['gene'])
            rows.append({'ranking': ranking, 'method': m, 'type': t, 'n': n, 'n_pbmc3k': len(a), 'n_pbmc10k': len(b),
                         'both': len(a & b), 'union': len(a | b), 'concordance': len(a & b) / max(len(a | b), 1),
                         'share_found_again': len(a & b) / max(len(a), 1)})
    return pd.DataFrame(rows)


def pooled(conc):
    """Per ranking, method and N, the sums over types: concordance and share found again; and the types whose
    top-5 lists share a gene."""
    g = conc.groupby(['ranking', 'method', 'n'])[['both', 'union', 'n_pbmc3k']].sum()
    out = pd.DataFrame({'concordance': g['both'] / g['union'], 'share_found_again': g['both'] / g['n_pbmc3k']})
    five = conc[conc['n'] == 5].assign(shared=lambda d: d['both'] > 0).groupby(['ranking', 'method'])['shared'].sum()
    return out.join(five.rename('types_sharing_top5'), on=['ranking', 'method']).reset_index()


def aucs(adata, genes, types):
    """Each type's AUC against the other included types for genes, U / (n1 n2) with ties counted one half."""
    x = adata[:, genes].layers['log1p_norm'].tocsc()
    labels = adata.obs['type'].to_numpy()
    out = pd.DataFrame(index=genes, columns=types, dtype=float)
    for start in range(0, len(genes), CHUNK):
        ranks = rankdata(x[:, start:start + CHUNK].toarray(), axis=0)
        for t in types:
            inside = labels == t
            n1, n2 = inside.sum(), (~inside).sum()
            out.iloc[start:start + CHUNK, types.index(t)] = (ranks[inside].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n2)
    return out


def effects(data, table, types):
    """Per type, the genes of the effect transfer and of the top-TOP_TRANSFER markers, present in both matrices, with
    LN's LFC, scanpy's LFC and the AUC in both datasets."""
    shared = data['PBMC3k'].var_names.intersection(data['PBMC10k'].var_names)
    top = table[(table['ranking'] == RANKING) & (table['rank'] <= TOP_TRANSFER)]
    nonzero = {name: (data[name][:, shared].layers['norm_counts'] > 0).astype(np.float32) for name in DATASETS}
    rows = []
    for t in types:
        passes = np.ones(len(shared), dtype=bool)
        for name in DATASETS:
            inside = (data[name].obs['type'] == t).to_numpy()
            x = nonzero[name]
            detected = np.maximum(np.asarray(x[inside].mean(0)).ravel(), np.asarray(x[~inside].mean(0)).ravel())
            passes &= detected >= DETECTED
        marked = {name: set(top[(top['dataset'] == name) & (top['type'] == t)]['gene']) & set(shared) for name in DATASETS}
        genes = sorted(set(shared[passes]) | marked['PBMC3k'] | marked['PBMC10k'])
        rows.append(pd.DataFrame({'type': t, 'gene': genes, 'transfer': np.isin(genes, shared[passes]),
                                  'marker_pbmc3k': np.isin(genes, list(marked['PBMC3k'])),
                                  'marker_pbmc10k': np.isin(genes, list(marked['PBMC10k']))}))
    df = pd.concat(rows, ignore_index=True)
    for name in DATASETS:
        genes = sorted(set(df['gene']))
        auc = aucs(data[name], genes, types)
        suffix = '_' + name.lower()
        for t in types:
            sel = df['type'] == t
            ln = _get_rank_genes_df(data[name], t, 'ln').set_index('names')['logfoldchanges']
            sc_lfc = _get_rank_genes_df(data[name], t, 't_test').set_index('names')['logfoldchanges']
            df.loc[sel, 'ln_lfc' + suffix] = ln.reindex(df.loc[sel, 'gene']).to_numpy(dtype=float)
            df.loc[sel, 'scanpy_lfc' + suffix] = sc_lfc.reindex(df.loc[sel, 'gene']).to_numpy(dtype=float)
            df.loc[sel, 'auc' + suffix] = auc.loc[df.loc[sel, 'gene'], t].to_numpy(dtype=float) - 0.5
    return df


def tls_slope(x, y):
    """The total-least-squares slope of y on x: the direction of their first principal axis."""
    v = np.linalg.eigh(np.cov(np.vstack([x, y])))[1][:, -1]
    return v[1] / v[0]


def transfer(df):
    """Per gene set (the transfer's genes, or each dataset's markers), measure and type (and pooled): Spearman
    correlation and total-least-squares slope of PBMC3k's value on PBMC10k's."""
    rows = []
    for subset in ('transfer', 'marker_pbmc3k', 'marker_pbmc10k'):
        d = df[df[subset]]
        for measure in MEASURES:
            for t, g in [('pooled', d), *d.groupby('type')]:
                x, y = g[f'{measure}_pbmc10k'].to_numpy(), g[f'{measure}_pbmc3k'].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                rows.append({'genes': subset, 'measure': measure, 'type': t, 'n': int(ok.sum()),
                             'not_finite': int((~ok).sum()), 'spearman': spearmanr(x[ok], y[ok])[0],
                             'tls_slope': tls_slope(x[ok], y[ok])})
    return pd.DataFrame(rows)


def draw_concordance(ax, pool, ranking=RANKING, column='concordance'):
    """Pooled concordance against N, one line per method."""
    d = pool[pool['ranking'] == ranking]
    x = np.arange(len(NS))
    for m in method_draw_order(METHODS):
        y = d[d['method'] == m].set_index('n').reindex(list(NS))[column]
        ax.plot(x, y, marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'],
                label=method_label(m), color=method_color(m))
    ax.set_xticks(x)
    ax.set_xticklabels(NS)
    ax.set_xlabel('Top N markers')
    ax.set_ylabel('Concordance' if column == 'concordance' else 'Fraction found again in PBMC10k')
    ax.set_ylim(0, 1)


def draw_by_type(ax, conc, n=N_MAIN, ranking=RANKING):
    """Each type's concordance at n, one point per method, side by side."""
    d = conc[(conc['ranking'] == ranking) & (conc['n'] == n)]
    types = sorted(d['type'].unique())
    x = np.arange(len(types))
    for k, m in enumerate(METHODS):
        y = d[d['method'] == m].set_index('type').reindex(types)['concordance']
        ax.plot(x + (k - 1) * 0.22, y, ls='none', marker='o', ms=plot_style.ERRORBAR['ms'], label=method_label(m),
                color=method_color(m))
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.set_ylabel(f'Concordance, top {n}')
    ax.set_ylim(0, 1)
    plot_style.boxplot_grid(ax)


def draw_transfer(ax, df, measure, subset='transfer', summary=None):
    """One measure's PBMC3k values against PBMC10k's, every type's genes together; the diagonal dashed, Spearman's
    rho and the total-least-squares slope written in."""
    d = df[df[subset]]
    x, y = d[f'{measure}_pbmc10k'].to_numpy(), d[f'{measure}_pbmc3k'].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    title, method = MEASURES[measure]
    ax.scatter(x[ok], y[ok], s=2, lw=0, alpha=0.4, color=method_color(method))
    lo, hi = np.nanmin(np.r_[x[ok], y[ok]]), np.nanmax(np.r_[x[ok], y[ok]])
    ax.plot([lo, hi], [lo, hi], color='0.4', lw=0.8, ls='--')
    if summary is not None:
        s = summary[(summary['genes'] == subset) & (summary['measure'] == measure) & (summary['type'] == 'pooled')].iloc[0]
        ax.text(0.04, 0.96, f'ρ = {s.spearman:.2f}\nslope = {s.tls_slope:.2f}', transform=ax.transAxes, va='top',
                fontsize=plot_style.SMALL)
    ax.set_title(title)
    ax.set_xlabel('PBMC10k')
    ax.set_ylabel('PBMC3k')


def save(fig, path, key=True):
    plt.tight_layout()
    if key:
        plot_style.legend_outside(fig)
    plot_style.rasterize_dense(fig)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot(conc, pool, df, summary):
    """The arm's figures: concordance against N (Fig 2d) and its sensitivities, and the effect transfer (Fig 2e)."""
    plot_style.use()
    for ranking in RANKINGS:
        fig, ax = plt.subplots(figsize=plot_style.figsize())
        draw_concordance(ax, pool, ranking)
        save(fig, OUT / fig_name('concordance' if ranking == RANKING else f'concordance_by_{ranking}'))
    fig, ax = plt.subplots(figsize=plot_style.figsize())
    draw_concordance(ax, pool, RANKING, 'share_found_again')
    save(fig, OUT / fig_name('share_found_again'))
    fig, ax = plt.subplots(figsize=plot_style.figsize(2))
    draw_by_type(ax, conc)
    save(fig, OUT / fig_name('concordance_by_type'))
    for subset, stem in (('transfer', 'effect_transfer'), ('marker_pbmc3k', 'effect_transfer_markers_pbmc3k'),
                         ('marker_pbmc10k', 'effect_transfer_markers_pbmc10k')):
        fig, axes = plt.subplots(1, len(MEASURES), figsize=plot_style.figsize(len(MEASURES)))
        for ax, measure in zip(axes, MEASURES):
            draw_transfer(ax, df, measure, subset, summary)
        save(fig, OUT / fig_name(stem), key=False)


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset concordance of each method\'s markers, PBMC3k and PBMC10k')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('markers', 'skipped', 'concordance', 'concordance_pooled', 'effects', 'transfer')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the replication {k}', source='this script without --plot-only'))
                  for k, p in csv.items()}
    else:
        data = {name: load(name) for name in DATASETS}
        for name, adata in data.items():
            print(f'  {name}: {adata.n_obs} cells, {adata.n_vars} genes; testing')
            run_tests(adata)
        types = sorted(data['PBMC3k'].obs['type'].cat.categories)
        universe = {'PBMC3k': annotation('PBMC10k'), 'PBMC10k': annotation('PBMC3k')}
        table, skipped = marker_table(data, universe, types)
        conc = concordance(table)
        df = effects(data, table, types)
        tables = {'markers': table, 'skipped': skipped, 'concordance': conc, 'concordance_pooled': pooled(conc),
                  'effects': df, 'transfer': transfer(df)}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['concordance'], tables['concordance_pooled'], tables['effects'], tables['transfer'])


if __name__ == '__main__':
    main()
