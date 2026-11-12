"""Planted effects in the memory CD4 T cells of PBMC10k, a figure outside the paper, assembled from the drawing code of
citeseq.plots.

The cells are those of the 10x 5' PBMC 10k CITE-seq run selected by surface protein (CD3+ CD4+ CD45RA-), with effects
planted in their RNA.
a, b: lfc_vs_true_all_methods and lfc_error_all_methods, the LFC estimates of LN, scanpy and Seurat for the planted
genes against the true LFC.
c: degs_summary, the TPR and FPR at Bonferroni 0.05, the AUPRC and the mean PR curves over the replicates, in one row
with its key.

    python -m figures.citeseq  # from reproducibility/, a few seconds; needs the citeseq results
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
