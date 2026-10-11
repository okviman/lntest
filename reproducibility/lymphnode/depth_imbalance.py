#!/usr/bin/env python3
"""Lymph node, the T-cell zone's reads thinned and the B-cell follicles whole: as the T-cell zone's variance grows at
equal means, do the baselines' calls leave the pseudo-bulk markers, and LN's not?

The follow-up to kidney/spot_split.py, where giving one half of a homogeneous population fewer 2 um spots opened a
variance difference at equal means. Here the groups are the B-cell follicles (Cluster-3) and the T-cell zone (Cluster-1),
the truth is each compartment's markers in the pseudo-bulk reference, and every read of the T-cell zone's 2 um spots is
kept with probability p in FRACTIONS (binomial thinning of each spot's counts; every spot kept, the follicles' reads all
kept), REPLICATES times per p, plus the unthinned data once. A T cell covers a median of 7 spots, so dropping spots would
shrink cells under the spot filter; dropping reads only lowers their counts, though below p 0.5 the 20-count filter
removes T cells (about 30% at 0.4, 52% at 0.3, 80% at 0.2), the deeper ones surviving. Cells, genes and the three tests are as lymphnode/
subsampling.py runs them (config_50rep.yaml: cells with at least min_spots_per_cell spots and
min_counts_after_subsampling counts, genes detected in at least min_cells_per_gene cells of each compartment;
Bonferroni, the Wilcoxon tie-corrected, LN on counts, which it normalises itself), and each compartment's upregulated
calls and full ranking are scored against its markers. Readouts per task, method and compartment: the median counts per
cell of each compartment and their ratio, the variance gap between them (clustering/multires_gaps.py's, on CP10K), the
calls, precision, recall and AUPRC, and for LN also the AUPRC of its marker rule's ranking (auprc_marker_rule);
summarised per condition, with the calls outside the reference markers.

With --thin bins, the spots themselves are subsampled instead, as the podocyte split does: every 2 um spot of the T-cell
zone is kept with probability p, and a T cell left under the filters (fewer than min_spots_per_cell spots or
min_counts_after_subsampling reads) gets dropped spots back, in random order, until it passes or has them all; below
p 0.4 most T cells sit at that floor.

    python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml               # from reproducibility/, minutes
    python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml --thin bins   # the spot version
    python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml --plot-only
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'   # one thread per worker, as lymphnode/subsampling.py

import argparse
import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from scipy import sparse

import plot_style
from clustering.multires_gaps import gaps
from de_utils import prepare_adata_layers, run_ln_de, run_ttest_de, run_wilcoxon_de
from evaluation_utils import _get_rank_genes_df, aupr_adata, aupr_from_ranking, get_all_sig_genes_adata
from lymphnode.subsampling import PVAL_THRESHOLD, load_config, precision_recall, reference_markers
from method_colors import method_color, method_draw_order, method_label, method_order
from paths import fig_name, require_input, results_dir

T_ZONE, B_FOLLICLES = 'Cluster-1', 'Cluster-3'
FRACTIONS = {'reads': (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 'bins': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
REPLICATES = 20
NAMES = {T_ZONE: 'T-cell zone', B_FOLLICLES: 'B-cell follicles'}
KEYS = {'LN test': 'ln_test', 't-test': 't_test', 'Wilcoxon': 'wilcoxon'}   # method labels as the arm's results name them
SEED = {None: 0, T_ZONE: 1, B_FOLLICLES: 3}
OUT = results_dir(__file__) / 'depth_imbalance'
OUTS = {'reads': OUT, 'bins': results_dir(__file__) / 'depth_imbalance_bins'}
XLABEL = {'reads': '$p_\\mathrm{sample}$', 'bins': '$p_\\mathrm{sub}$'}   # the T-cell zone's reads kept, as the kidney's
# UMI downsampling (p_sample), or its spots kept, as equal thinning's (p_sub)
_W = {}


def init(adata_path, cluster_column, cell_id_column, markers, filters, thin='reads'):
    """Each worker's copy of the spots, their cells and compartments, the markers, the filters and how to thin."""
    warnings.filterwarnings('ignore')
    a = sc.read_h5ad(adata_path)
    cells, spot_cell = np.unique(a.obs[cell_id_column].astype(str).str.strip().to_numpy(), return_inverse=True)
    spot_cluster = a.obs[cluster_column].astype(str).str.strip().to_numpy()
    cell_cluster = np.empty(len(cells), dtype=object)
    cell_cluster[spot_cell] = spot_cluster
    X = sparse.csr_matrix(a.X, dtype=np.float32)
    if not np.array_equal(X.data, np.round(X.data)):
        raise SystemExit('the spot matrix is not counts; binomial thinning needs integers')
    order = np.argsort(spot_cell, kind='stable')
    cell_spots = np.split(order, np.cumsum(np.bincount(spot_cell, minlength=len(cells)))[:-1])
    _W.update(X=X, var_names=a.var_names.astype(str).to_numpy(), cells=cells, spot_cell=spot_cell,
              spot_cluster=spot_cluster, cell_cluster=cell_cluster, markers=markers, thin=thin, cell_spots=cell_spots,
              spot_reads=np.asarray(X.sum(1)).ravel(), **filters)


def aggregate(X, keep=None):
    """Cell-level counts from the kept spots of X (all if keep is None), filtered as lymphnode/subsampling.py filters
    them."""
    n_cells, n_spots = len(_W['cells']), X.shape[0]
    idx = np.arange(n_spots) if keep is None else np.flatnonzero(keep)
    spots = np.bincount(_W['spot_cell'][idx], minlength=n_cells)
    a = sparse.csr_matrix((np.ones(len(idx), dtype=np.float32), (_W['spot_cell'][idx], idx)),
                          shape=(n_cells, n_spots))
    counts = (a @ X).tocsr()
    counts.eliminate_zeros()
    ok = spots >= _W['min_spots_per_cell']
    adata = sc.AnnData(X=counts[ok], obs=pd.DataFrame({'Cluster': pd.Categorical(_W['cell_cluster'][ok])},
                                                       index=_W['cells'][ok]),
                       var=pd.DataFrame(index=_W['var_names']))
    for _ in range(50):
        before = adata.shape
        keep_genes = np.ones(adata.n_vars, dtype=bool)
        for cl in adata.obs['Cluster'].cat.categories:
            mask = (adata.obs['Cluster'] == cl).to_numpy()
            keep_genes &= np.asarray((adata.X[mask] > 0).sum(0)).ravel() >= _W['min_cells_per_gene']
        adata = adata[:, keep_genes].copy()
        adata = adata[np.asarray(adata.X.sum(1)).ravel() >= _W['min_counts_after_subsampling']].copy()
        if adata.shape == before or 0 in adata.shape:
            break
    adata.obs['Cluster'] = pd.Categorical(adata.obs['Cluster'].astype(str))
    return adata


def thin(thinned, p, rng):
    """The spot counts with every read of the thinned compartment's spots kept with probability p."""
    X = _W['X']
    if thinned is None:
        return X
    on = _W['spot_cluster'][np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))] == thinned
    data = X.data.copy()
    data[on] = rng.binomial(data[on].astype(np.int64), p)
    out = sparse.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


def subsample_spots(thinned, p, rng):
    """Every spot of the thinned compartment kept with probability p; a cell of it left under the filters gets its
    dropped spots back, in random order, until it passes or has them all."""
    keep = np.ones(_W['X'].shape[0], dtype=bool)
    if thinned is None:
        return keep
    t = _W['spot_cluster'] == thinned
    keep[t] = rng.random(int(t.sum())) < p
    cell, reads, n_cells = _W['spot_cell'], _W['spot_reads'], len(_W['cells'])
    n = np.bincount(cell[keep], minlength=n_cells)
    r = np.bincount(cell[keep], weights=reads[keep], minlength=n_cells)
    short = (n < _W['min_spots_per_cell']) | (r < _W['min_counts_after_subsampling'])
    for c in np.flatnonzero(short & (_W['cell_cluster'] == thinned)):
        spots = _W['cell_spots'][c]
        for s in rng.permutation(spots[~keep[spots]]):
            if n[c] >= _W['min_spots_per_cell'] and r[c] >= _W['min_counts_after_subsampling']:
                break
            keep[s] = True
            n[c] += 1
            r[c] += reads[s]
    return keep


def tested(thinned, p, replicate):
    """One condition and replicate's cells, with the three tests run: the thinned compartment (None for none), p, the
    replicate."""
    rng = np.random.default_rng([SEED[thinned], int(round(p * 100)), replicate])
    if _W['thin'] == 'reads':
        adata = aggregate(thin(thinned, p, rng))
    else:
        adata = aggregate(_W['X'], subsample_spots(thinned, p, rng))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepare_adata_layers(adata)
        run_ln_de(adata, 'Cluster', key_added='ln_test', layer='counts', sparse=True)
        run_ttest_de(adata, 'Cluster', key_added='t_test', layer='log1p_norm', use_raw=False)
        run_wilcoxon_de(adata, 'Cluster', key_added='wilcoxon', layer='log1p_norm', use_raw=False)
    return adata


def moments(cp):
    """Each gene's CP10K mean, variance and number of detecting cells."""
    n = cp.shape[0]
    mean = np.asarray(cp.mean(0)).ravel()
    var = (np.asarray(cp.multiply(cp).sum(0)).ravel() - n * mean ** 2) / max(n - 1, 1)
    return mean, var, np.asarray((cp > 0).sum(0)).ravel()


def marker_rule_ranking(adata, cl, lab, det):
    """LN's ranking as its marker rule (clustering.marker_rule.detection_ln): its significant up calls ordered by the
    detection score, log((a1 / n1) / (a0 / n0)) over its Katz SE on the cells detecting each gene, then every other
    gene by LN's score."""
    own = lab == cl
    n1, n0 = own.sum(), (~own).sum()
    a1 = np.maximum(np.asarray(det[own].sum(0)).ravel().astype(float), 1)
    a0 = np.maximum(np.asarray(det[~own].sum(0)).ravel().astype(float), 1)
    z = pd.Series(np.log((a1 / n1) / (a0 / n0)) / np.sqrt(1 / a1 - 1 / n1 + 1 / a0 - 1 / n0), index=adata.var_names)
    df = _get_rank_genes_df(adata, cl, KEYS['LN test']).dropna(subset=['scores'])
    df = df.assign(names=df['names'].astype(str), scores=df['scores'].astype(float))
    called = (df['pvals_adj'] < PVAL_THRESHOLD) & (df['scores'] > 0)
    first = df[called].assign(z=z.reindex(df.loc[called, 'names']).to_numpy())
    first = first.sort_values(['z', 'scores', 'names'], ascending=[False, False, True])['names'].tolist()
    rest = df[~called].sort_values(['scores', 'names'], ascending=[False, True])['names'].tolist()
    order = first + rest
    return order, np.arange(len(order), 0, -1, dtype=float)


def score(adata, thinned, p, replicate):
    """Each compartment's and method's calls, precision, recall and AUPRC, with the compartments' depths and the
    variance gap between them."""
    depth = np.asarray(adata.layers['counts'].sum(1)).ravel()
    lab = adata.obs['Cluster'].astype(str).to_numpy()
    med = {c: float(np.median(depth[lab == c])) for c in (T_ZONE, B_FOLLICLES)}
    cp = sparse.csr_matrix(adata.layers['norm_counts'], dtype=np.float64)
    gap = gaps(*moments(cp[lab == B_FOLLICLES]), *moments(cp[lab == T_ZONE]))
    tested = set(adata.var_names)
    det = (sparse.csr_matrix(adata.layers['counts']) > 0).tocsr()
    rows = []
    for cl in (T_ZONE, B_FOLLICLES):
        markers = set(_W['markers'][cl])
        rule = aupr_from_ranking(*marker_rule_ranking(adata, cl, lab, det), markers)
        for method, key in KEYS.items():
            sig = get_all_sig_genes_adata(adata, cl, key=key, pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
            prec, rec = precision_recall(sig, markers, tested)
            rows.append({'thinned': thinned or 'none', 'p': p, 'replicate': replicate, 'cluster': cl, 'method': method,
                         'calls': len(sig), 'precision': prec, 'recall': rec,
                         'auprc': aupr_adata(adata, cl, markers, key=key), 'depth_t': med[T_ZONE],
                         'depth_b': med[B_FOLLICLES], 'depth_ratio': med[B_FOLLICLES] / med[T_ZONE],
                         'variance_gap': gap['gap_equal_mean'], 'cells': int((lab == cl).sum()), 'genes': adata.n_vars,
                         'auprc_marker_rule': rule if method == 'LN test' else np.nan})
    return rows


def task(args):
    """One condition and replicate: the thinned compartment (None for none), p, the replicate."""
    return score(tested(*args), *args)


def tasks(replicates, thin='reads'):
    """The unthinned data once, then the T-cell zone thinned at each fraction, replicates times."""
    return [(None, 1.0, 0)] + [(T_ZONE, p, r) for p in FRACTIONS[thin] for r in range(replicates)]


def run(config, n_workers, thin='reads'):
    """Every condition and replicate, in a fixed order."""
    markers = reference_markers(config['reference'])
    filters = {k: config[k] for k in ('min_spots_per_cell', 'min_counts_after_subsampling', 'min_cells_per_gene')}
    adata_path = require_input(config['adata_path'], what='the Visium HD lymph node matrix',
                               source='bash lymphnode/reproduce_vishd_cluster1_cluster3.sh')
    todo = tasks(REPLICATES, thin)
    ctx = multiprocessing.get_context('spawn')
    # a fresh worker per task: a worker's earlier tasks changed one Wilcoxon result between runs
    with ctx.Pool(n_workers, initializer=init, maxtasksperchild=1,
                  initargs=(str(adata_path), config['cluster_column'], config['cell_id_column'], markers, filters,
                            thin)) as pool:
        rows = []
        for k, r in enumerate(pool.imap(task, todo, chunksize=1)):
            rows += r
            if (k + 1) % 50 == 0:
                print(f'  {k + 1}/{len(todo)} tasks', flush=True)
    return pd.DataFrame(rows)


def false_calls(df):
    """Each row's calls outside the compartment's reference markers, from its calls and precision."""
    return (df['calls'] - (df['calls'] * df['precision'].fillna(0)).round()).astype(int)


def summarise(df):
    """Means over replicates per condition, method and compartment, with the calls outside the reference markers and
    the share of the compartment's unthinned cells kept; and, per method, least-squares slopes against log2 of the B/T
    depth ratio of the recall gap (B minus T) and of each compartment's precision."""
    g = df.assign(false_calls=false_calls(df)).groupby(['thinned', 'p', 'method', 'cluster'])
    s = g[['calls', 'false_calls', 'precision', 'recall', 'auprc', 'depth_ratio', 'variance_gap', 'cells']].mean()
    s = s.reset_index()
    full = s[s['thinned'] == 'none'].drop_duplicates('cluster').set_index('cluster')['cells']
    s['kept'] = s['cells'] / s['cluster'].map(full)
    wide = s.pivot_table(index=['thinned', 'p', 'method'], columns='cluster',
                         values=['recall', 'precision', 'depth_ratio']).reset_index()
    wide.columns = ['_'.join(c).strip('_') for c in wide.columns]
    wide['log2_ratio'] = np.log2(wide[f'depth_ratio_{B_FOLLICLES}'])
    wide['recall_gap'] = wide[f'recall_{B_FOLLICLES}'] - wide[f'recall_{T_ZONE}']
    slopes = []
    for m, d in wide.groupby('method'):
        row = {'method': m, 'recall_gap_slope': np.polyfit(d['log2_ratio'], d['recall_gap'], 1)[0]}
        for cl in (T_ZONE, B_FOLLICLES):
            ok = d[f'precision_{cl}'].notna()
            row[f'precision_slope_{cl}'] = np.polyfit(d.loc[ok, 'log2_ratio'], d.loc[ok, f'precision_{cl}'], 1)[0]
        slopes.append(row)
    return s, wide, pd.DataFrame(slopes)


def design_path(d):
    """One condition per row in design order: any B-cell follicle subsampling from the lowest p up, the unthinned
    data, then the T-cell zone subsampled down to its lowest p."""
    return pd.concat([d[d['thinned'] == B_FOLLICLES].sort_values('p'), d[d['thinned'] == 'none'],
                      d[d['thinned'] == T_ZONE].sort_values('p', ascending=False)])


def draw_against_ratio(ax, wide, column, ylabel):
    """One summary column against log2 of the B/T depth ratio, one colour per method; subsampling the T-cell zone in
    filled points, the B-cell follicles in open ones, the unthinned data as a cross."""
    for m in method_draw_order(list(KEYS)):
        d = wide[wide['method'] == m]
        path = design_path(d)
        ax.plot(path['log2_ratio'], path[column], color=method_color(m), lw=0.8, label=method_label(m))
        for thinned, style in ((T_ZONE, dict(mfc=method_color(m))), (B_FOLLICLES, dict(mfc='white')),
                               ('none', dict(marker='x'))):
            e = d[d['thinned'] == thinned]
            ax.plot(e['log2_ratio'], e[column], ls='none', marker=style.get('marker', 'o'), ms=3,
                    mfc=style.get('mfc', method_color(m)), color=method_color(m))
    ax.axvline(0, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('Depth ratio, B / T (log2)')
    ax.set_ylabel(ylabel)


def draw_errorbar(ax, df, column, ylabel, xlabel=XLABEL['reads']):
    """A column's mean over each condition's tasks against the fraction p of the T-cell zone kept (1, the unthinned
    data), with bars of one SD, one colour per method, as Fig 1's error bars."""
    for m in method_draw_order(list(KEYS)):
        d = df[df['method'] == m].groupby('p')[column].agg(['mean', 'std'])
        ax.errorbar(d.index, d['mean'], yerr=d['std'].fillna(0), color=method_color(m), label=method_label(m),
                    **plot_style.ERRORBAR)
    if column == 'auprc':
        ax.set_ylim(0, 1)   # as every AUPRC in the paper
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def per_task_false_calls(df):
    """Each task's and method's calls outside the reference markers, both compartments together, with its depth
    ratio."""
    return df.assign(false_calls=false_calls(df)).groupby(['thinned', 'p', 'replicate', 'method'], as_index=False).agg(
        false_calls=('false_calls', 'sum'), depth_ratio=('depth_ratio', 'first'))


def method_lines():
    """The method key of the error-bar figures, as Fig 1's."""
    return [Line2D([], [], color=method_color(m), marker='o', ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'],
                   label=method_label(m)) for m in method_order(list(KEYS))]


def save_with_key(fig, stem, out):
    """An error-bar figure with the method key right of it."""
    lines = method_lines()
    plot_style.legend_outside(fig, lines, [h.get_label() for h in lines])
    path = out / fig_name(stem)
    fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot(wide, df, thin='reads'):
    """The arm's figures, one plot each: each compartment's recall, precision and the recall gap against the ratio;
    each compartment's AUPRC and the calls outside the reference markers per condition."""
    plot_style.use()
    out = OUTS[thin]
    panels = [('recall_gap', 'Recall, B-cell follicles\nminus T-cell zone', 'depth_imbalance_recall_gap')]
    for cl in (T_ZONE, B_FOLLICLES):
        short = 'b_follicles' if cl == B_FOLLICLES else 't_zone'
        panels += [(f'recall_{cl}', f'Recall, {NAMES[cl]}', f'depth_imbalance_recall_{short}'),
                   (f'precision_{cl}', f'Precision, {NAMES[cl]}', f'depth_imbalance_precision_{short}')]
    for column, ylabel, stem in panels:
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_against_ratio(ax, wide, column, ylabel)
        plot_style.legend_outside(fig)
        path = out / fig_name(stem)
        fig.savefig(path, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {path}')
    for cl in (T_ZONE, B_FOLLICLES):
        fig, ax = plt.subplots(figsize=plot_style.PANEL)
        draw_errorbar(ax, df[df['cluster'] == cl], 'auprc', f'AUPRC, markers of\nthe {NAMES[cl]}', XLABEL[thin])
        save_with_key(fig, f"depth_imbalance_auprc_{'b_follicles' if cl == B_FOLLICLES else 't_zone'}", out)
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_errorbar(ax, per_task_false_calls(df), 'false_calls', 'Calls outside the\nreference markers', XLABEL[thin])
    save_with_key(fig, 'depth_imbalance_false_calls', out)


def main():
    parser = argparse.ArgumentParser(description="The lymph node's T-cell zone reads thinned, the follicles whole")
    parser.add_argument('--config', required=True, help='The subsampling config (lymphnode/config_50rep.yaml)')
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--thin', choices=list(OUTS), default='reads', help='Thin the T-cell zone\'s reads or its spots')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    out = OUTS[args.thin]
    out.mkdir(parents=True, exist_ok=True)
    csv = {k: out / f'{k}.csv' for k in ('results', 'summary', 'by_ratio', 'slopes')}
    if args.plot_only:
        wide, df = (pd.read_csv(require_input(csv[k], what=f'the depth-imbalance {k}',
                                              source='this script without --plot-only'))
                    for k in ('by_ratio', 'results'))
    else:
        df = run(load_config(args.config), args.n_workers, args.thin)
        df = df.sort_values(['thinned', 'p', 'replicate', 'cluster', 'method'], ignore_index=True)
        summary, wide, slopes = summarise(df)
        for k, t in (('results', df), ('summary', summary), ('by_ratio', wide), ('slopes', slopes)):
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
        cols = ['thinned', 'p', 'method', 'cluster', 'depth_ratio', 'variance_gap', 'kept', 'false_calls', 'auprc']
        print(summary[cols].round(3).to_string(index=False))
    plot(wide, df, args.thin)


if __name__ == '__main__':
    main()
