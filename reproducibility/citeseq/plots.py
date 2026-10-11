"""Every CITE-seq figure, from the CSVs that exp.py and the R chain write; nothing is drawn in R.

    python -m citeseq.plots      # from reproducibility/
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score

import plot_style
from method_colors import MEDIANPROPS, method_color, method_draw_order, method_label, method_order
from paths import fig_name, require_input, results_dir

RESULTS = results_dir(__file__)
GATE_SHADES = {'-': '0.8', '+': '0.5', '++': '0.2'}
# The metrics of the manuscript's metrics table (RECOMB Table 3), in its order, and FPR beside TNR
TABLE_METRICS = {'accuracy': 'Accuracy', 'precision': 'Precision', 'tpr': 'TPR', 'tnr': 'TNR', 'fpr': 'FPR', 'f1': 'F1'}
# Scanpy's and Seurat's log1p t-tests appear side by side here, so both are named by tool
SCANPY_LOG1P = f"Scanpy {method_label('t-test')}"


def save(fig, stem, verbose=True):
    out = RESULTS / fig_name(stem)
    plot_style.rasterize_dense(fig)
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f'wrote {out}')


def gating_subsets():
    """Each gating marker with the cells it is gated on: CD3 and CD4 on all cells, CD45RA within CD3+ CD4++."""
    df = pd.read_csv(require_input(RESULTS / 'adt_gating.csv',
                                   what='CLR values and GMM gates of the PBMC10k ADTs',
                                   source='cd citeseq/R && Rscript pbmc10k_process.R'))
    cd4_t = df[(df.CD3_cluster == '+') & (df.CD4_cluster == '++')]  # CD45RA is gated within these
    return (('CD3', df), ('CD4', df), ('CD45RA', cd4_t))


def draw_gating(ax, marker, sub):
    """CLR density of one gating marker, one fill per GMM gate."""
    grid = np.linspace(sub[marker].min(), sub[marker].max(), 400)
    for gate in [g for g in GATE_SHADES if g in set(sub[f'{marker}_cluster'])]:
        x = sub.loc[sub[f'{marker}_cluster'] == gate, marker]
        ax.fill_between(grid, gaussian_kde(x, bw_method='silverman')(grid), color=GATE_SHADES[gate],
                        alpha=0.7, lw=0, label=f"{marker}{gate.replace('-', '−')}")
    ax.set_xlabel(f'{marker} (CLR)')
    ax.set_ylabel('Density')
    ax.set_title(f'{marker} CLR distribution with GMM clustering')


def gating():
    """CLR density of each gating marker, one fill per GMM gate."""
    for marker, sub in gating_subsets():
        fig, ax = plt.subplots(figsize=plot_style.figsize())
        draw_gating(ax, marker, sub)
        fig.tight_layout()
        plot_style.legend_outside(fig)
        save(fig, f'gating_{marker.lower()}')


def annotate_r2(ax, true_lfc, est):
    """Squared Pearson correlation of estimated and true LFC over the plotted genes, top left."""
    true_lfc, est = np.asarray(true_lfc), np.asarray(est)
    ok = np.isfinite(true_lfc) & np.isfinite(est)
    r = np.corrcoef(true_lfc[ok], est[ok])[0, 1]
    ax.text(0.14, 0.84, f'$R^2$ = {r ** 2:.3f}', transform=ax.transAxes, ha='left', va='top')  # the figure's text size


def draw_scatter(axes, true_lfc, panels, error=False):
    """Estimated LFC (or its error) against the true LFC, one axes per estimator."""
    for ax, (title, color, est) in zip(axes, panels):
        ax.scatter(true_lfc, true_lfc - est if error else est, s=2, color=color, alpha=0.5, lw=0)
        if error:
            ax.axhline(0, color='black', ls='--', lw=0.8)
        else:
            lo, hi = np.nanmin(true_lfc), np.nanmax(true_lfc)
            ax.plot([lo, hi], [lo, hi], color='black', ls='--', lw=0.8)
            annotate_r2(ax, true_lfc, est)
        ax.set_title(title)
        ax.set_xlabel('True LFC')
    axes[0].set_ylabel('Bias' if error else 'Estimated LFC')


def scatter_panels(true_lfc, panels, stem, error=False, verbose=True):
    """Estimated LFC (or its error) against the true LFC, one panel per estimator."""
    fig, axes = plt.subplots(1, len(panels), figsize=plot_style.figsize(len(panels)), sharex=True, sharey=True)
    draw_scatter(axes, true_lfc, panels, error)
    fig.tight_layout()
    save(fig, stem, verbose)


def all_methods(res, seurat):
    """The true LFCs of the planted genes Seurat also tested, and LN's, Scanpy's and Seurat's estimates of them."""
    merged = res.rename_axis('gene_names').reset_index().rename(columns={'replicate': 'rep'}).merge(
        seurat[['gene_names', 'seurat_lfc', 'rep']], on=['gene_names', 'rep'])
    merged = merged[merged.true_lfc != 0]
    panels = [(method_label('LN'), method_color('LN'), merged.ln_lfc),
              (SCANPY_LOG1P, method_color('t-test'), merged.scanpy_lfc),
              (method_label('seurat'), method_color('seurat'), merged.seurat_lfc)]
    return merged.true_lfc, panels


def lfc():
    res = pd.read_csv(require_input(RESULTS / 'CITE_seq_results.csv', what='per-gene LFCs of every replicate',
                                    source='python -m citeseq.exp'), index_col=0)
    sig = res[res.is_signal_gene]
    panels = [(method_label('LN'), method_color('LN'), sig.ln_lfc),
              (SCANPY_LOG1P, method_color('t-test'), sig.scanpy_lfc)]
    scatter_panels(sig.true_lfc, panels, 'lfc_vs_true')
    scatter_panels(sig.true_lfc, panels, 'lfc_error', error=True)

    seurat_csv = RESULTS / 'seurat_lfc_results.csv'
    if not seurat_csv.exists():
        print(f'skipped the Seurat figures: no {seurat_csv} (cd citeseq/R && Rscript pbmc10k_seurat.R)')
        return
    seurat = pd.read_csv(seurat_csv)
    fig, ax = plt.subplots(figsize=plot_style.figsize())
    scatter = seurat[seurat.true_lfcs != 0]
    ax.scatter(scatter.true_lfcs, scatter.seurat_lfc, s=2, color=method_color('seurat'), alpha=0.5, lw=0)
    lo, hi = scatter.true_lfcs.min(), scatter.true_lfcs.max()
    ax.plot([lo, hi], [lo, hi], color='black', ls='--', lw=0.8)
    annotate_r2(ax, scatter.true_lfcs, scatter.seurat_lfc)
    ax.set_xlabel('True LFC')
    ax.set_ylabel('Estimated LFC')
    ax.set_title(method_label('seurat'))
    fig.tight_layout()
    save(fig, 'lfc_vs_true_seurat')

    true_lfc, panels = all_methods(res, seurat)
    scatter_panels(true_lfc, panels, 'lfc_vs_true_all_methods')
    scatter_panels(true_lfc, panels, 'lfc_error_all_methods', error=True)


def ranking_curve(is_de, adj_p):
    """Precision and recall down the ranking by adjusted p-value (the kidney arm's, kidney/spot_split.py
    de_test_single), and the AUPRC as everywhere in the paper: sklearn's average precision of that ranking."""
    score = -np.log10(np.clip(np.nan_to_num(adj_p, nan=1.0), 1e-300, 1.0))
    hits = is_de[np.argsort(score)[::-1]]
    tp = np.cumsum(hits)
    precision = np.concatenate([[1.0], tp / np.arange(1, hits.size + 1)])
    recall = np.concatenate([[0.0], tp / hits.sum()])
    return precision, recall, average_precision_score(is_de, score)


def box(ax, k, values, method, rng):
    bp = ax.boxplot([values], positions=[k], widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=MEDIANPROPS)
    bp['boxes'][0].set_facecolor(method_color(method))
    bp['boxes'][0].set_edgecolor('none')
    plot_style.strip(ax, k, values, method_color(method), 0.6, rng)


def method_ticks(ax, methods):
    """Method names under their boxes, "t-test" on a second line so neighbours do not touch."""
    ax.set_xticks(range(len(methods)), [method_label(m).replace(' $t$', '\n$t$') for m in methods])


def metrics():
    """The metrics table as box plots: one panel per metric, one box per method, a point per replicate."""
    df = pd.read_csv(require_input(RESULTS / 'CITE_seq_metrics.csv', what='per-replicate DE metrics',
                                   source='python -m citeseq.exp'), index_col=0)
    methods = method_order([c for c in df.columns if c != 'replicate'])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=plot_style.figsize(3, 2))
    for ax, (key, name) in zip(axes.flat, TABLE_METRICS.items()):
        rows = df.loc[key]
        zero = plot_style.zero_row(rows[methods].to_numpy().ravel(), 0) if key == 'fpr' else None
        for k, m in enumerate(methods):
            values = rows[m].to_numpy()
            box(ax, k, np.where(values > 0, values, zero) if zero else values, m, rng)
        if zero:
            plot_style.log_axis_with_zero(ax, zero, rows[methods].to_numpy().max())
        method_ticks(ax, methods)
        ax.set_ylabel(name)
        plot_style.boxplot_grid(ax)
    fig.tight_layout()
    plot_style.legend_outside(fig, [Patch(facecolor=method_color(m)) for m in methods],
                              [method_label(m) for m in methods])
    save(fig, 'degs_metrics')


def summary_data():
    """Per replicate and method: TPR and FPR at Bonferroni 0.05 and AUPRC; and each method's PR curves on a recall grid."""
    res = pd.read_csv(require_input(RESULTS / 'CITE_seq_results.csv', what='per-gene results of every replicate',
                                    source='python -m citeseq.exp'), index_col=0)
    columns = {'LN': 'ln_adj_pvalue', 't-test': 't_adj_pvalue', 'Wilcoxon': 'w_adj_pvalue'}
    methods = method_order(columns)
    grid = np.linspace(0, 1, 101)
    rows, curves = [], {m: [] for m in methods}
    for _, d in res.groupby('replicate'):
        is_de = d['is_signal_gene'].to_numpy(bool)
        for m in methods:
            adj_p = d[columns[m]].to_numpy()
            called = adj_p < 0.05
            precision, recall, area = ranking_curve(is_de, adj_p)
            curves[m].append(np.interp(grid, recall, precision, left=precision[0], right=precision[-1]))
            rows.append({'method': m, 'TPR': called[is_de].mean(), 'FPR': called[~is_de].mean(), 'AUPRC': area})
    return pd.DataFrame(rows), curves, grid, methods


def draw_summary(axes, df, curves, grid, methods):
    """The summary's four panels into a 2 x 2 array of axes; returns the method key."""
    rng = np.random.default_rng(0)
    zero = plot_style.zero_row(df['FPR'], np.zeros(len(df)))
    for ax, metric in zip(axes.flat[:3], ['TPR', 'FPR', 'AUPRC']):
        for k, m in enumerate(methods):
            values = df.loc[df['method'] == m, metric].to_numpy()
            box(ax, k, np.where(values > 0, values, zero) if metric == 'FPR' else values, m, rng)
        method_ticks(ax, methods)
        ax.set_ylabel(metric)
        plot_style.boxplot_grid(ax)
        if metric != 'FPR':
            ax.set_ylim(0, 1)
    plot_style.log_axis_with_zero(axes[0, 1], zero, df['FPR'].max())

    ax = axes[1, 1]
    for m in method_draw_order(methods):
        c = np.array(curves[m])
        mean, sd = c.mean(0), c.std(0)
        ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=method_color(m),
                        alpha=0.15, lw=0)
        ax.plot(grid, mean, color=method_color(m), lw=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return [Patch(facecolor=method_color(m)) for m in methods], [method_label(m) for m in methods]


def summary():
    """The kidney with-DEG summary's panels for this experiment's single, balanced condition:
    TPR and FPR at Bonferroni 0.05, AUPRC, and the mean PR curves, over the replicates."""
    df, curves, grid, methods = summary_data()
    fig, axes = plt.subplots(2, 2, figsize=plot_style.figsize(2, 2))
    key = draw_summary(axes, df, curves, grid, methods)
    fig.tight_layout()
    plot_style.legend_outside(fig, *key)
    save(fig, 'degs_summary')
    return df


if __name__ == '__main__':
    plot_style.use()
    gating()
    lfc()
    metrics()
    summary()
