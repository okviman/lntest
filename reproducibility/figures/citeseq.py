"""Planted effects in PBMC10k's memory CD4 T cells, unnumbered (Extended Data Fig 8 until 2026-10-08) (the 10x 5' PBMC 10k CITE-seq run, the cells
selected by surface protein, CD3+ CD4+ CD45RA-, the effects planted in their RNA), assembled from the citeseq arm's
drawing code.

a: lfc_vs_true_all_methods, b: lfc_error_all_methods, LN's, Scanpy's and Seurat's LFC estimates of the planted genes
against the true LFC; c: degs_summary, TPR and FPR at Bonferroni 0.05, AUPRC and the mean PR curves over the replicates,
in one row with its key, the boxes named by colour only (citeseq.plots). Each is drawn by its own drawing code from the saved results, at the size it
prints at; a change to a panel belongs in that code.

    python -m figures.citeseq  # from reproducibility/, a few seconds; needs citeseq's results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from citeseq import plots as cs
from figures.layout import KEY_W, PLOT, ED_WIDTH as WIDTH, assemble, axes_at, save, use
from paths import require_input

TITLE = 0.25                      # space above axes that carry a title
SCATTER_H, SCATTER_GAP = 1.75, 0.12   # a, b: height, and the gap between axes that share y
SUMMARY_H, SUMMARY_LEFT = 1.65, 0.5   # c: height, and room for the y labels


def scatter_row(true_lfc, panels, error, stem):
    """a or b: one axes per estimator, sharing both axes, y labelled on the first only, as lfc_*_all_methods."""
    n = len(panels)
    aw = (WIDTH - PLOT['left'] - PLOT['right'] - (n - 1) * SCATTER_GAP) / n
    ah = SCATTER_H - PLOT['bottom'] - TITLE
    fig = plt.figure(figsize=(WIDTH, SCATTER_H))
    axes = []
    for k in range(n):
        share = {'sharex': axes[0], 'sharey': axes[0]} if axes else {}
        axes.append(axes_at(fig, PLOT['left'] + k * (aw + SCATTER_GAP), PLOT['bottom'], aw, ah, **share))
    cs.draw_scatter(axes, true_lfc, panels, error)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    return save(fig, stem)


def summary_row(stem):
    """c: degs_summary's four panels in one row, with its key."""
    df, curves, grid, methods = cs.summary_data()
    slot = (WIDTH - KEY_W) / 4
    aw, ah = slot - SUMMARY_LEFT - PLOT['right'], SUMMARY_H - PLOT['bottom'] - PLOT['top']
    fig = plt.figure(figsize=(WIDTH, SUMMARY_H))
    row = [axes_at(fig, k * slot + SUMMARY_LEFT, PLOT['bottom'], aw, ah) for k in range(4)]
    plot_style.legend_outside(fig, *cs.draw_summary(np.array(row).reshape(2, 2), df, curves, grid, methods))
    for ax in row[:3]:
        ax.tick_params(labelbottom=False)   # too narrow for the method names; the key names the colours
    return save(fig, stem)


if __name__ == '__main__':
    use()
    res = pd.read_csv(require_input(cs.RESULTS / 'CITE_seq_results.csv', what='per-gene LFCs of every replicate',
                                    source='python -m citeseq.exp'), index_col=0)
    seurat = pd.read_csv(require_input(cs.RESULTS / 'seurat_lfc_results.csv', what="Seurat's LFCs",
                                       source='cd citeseq/R && Rscript pbmc10k_seurat.R'))
    true_lfc, panels = cs.all_methods(res, seurat)
    a = scatter_row(true_lfc, panels, False, 'citeseq_a_lfc_vs_true_all_methods')
    b = scatter_row(true_lfc, panels, True, 'citeseq_b_lfc_error_all_methods')
    c = summary_row('citeseq_c_degs_summary')
    assemble([[('a', a, 0)], [('b', b, 0)], [('c', c, 0)]], 'citeseq', width=WIDTH)
