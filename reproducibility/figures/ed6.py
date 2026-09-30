"""Extended Data Fig 6: CITE-seq detection, assembled from the arm's existing figure.

a: degs_summary with its key (citeseq.plots): TPR and FPR at Bonferroni 0.05, PR-AUC and the mean PR curves over the
replicates. Drawn by its own drawing code from the saved results, at the size it prints at; a change to the panel
belongs in that code.

    python -m figures.ed6      # from reproducibility/, a few seconds; needs citeseq's results
"""

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from citeseq import plots as cs
from figures.layout import ED_WIDTH as WIDTH
from figures.layout import KEY_W, PLOT, assemble, axes_at, save, use

SUMMARY_H, SUMMARY_BOTTOM = 1.65, 0.5  # each row's height, and room for the two-line method names


def summary_panel():
    """a: degs_summary's four panels, two by two, with its key."""
    df, curves, grid, methods = cs.summary_data()
    slot = (WIDTH - KEY_W) / 2
    aw, ah = slot - PLOT['left'] - PLOT['right'], SUMMARY_H - SUMMARY_BOTTOM - PLOT['top']
    fig = plt.figure(figsize=(WIDTH, 2 * SUMMARY_H))
    axes = np.array([[axes_at(fig, c * slot + PLOT['left'], (1 - r) * SUMMARY_H + SUMMARY_BOTTOM, aw, ah)
                      for c in range(2)] for r in range(2)])
    plot_style.legend_outside(fig, *cs.draw_summary(axes, df, curves, grid, methods))
    return save(fig, 'ed6_a_degs_summary')


if __name__ == '__main__':
    use()
    assemble([[('a', summary_panel(), 0)]], 'ed6', width=WIDTH)
