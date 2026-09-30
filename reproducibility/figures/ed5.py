"""Extended Data Fig 5: CITE-seq, assembled from the arm's existing figures.

a-c: gating_cd3, gating_cd4, gating_cd45ra, each with its key; d: lfc_vs_true_all_methods; e: lfc_error_all_methods
(citeseq.plots). degs_summary is Extended Data Fig 6 (figures/ed6.py). Each is drawn by its own drawing code from the saved results, at the
size it prints at; a change to a panel belongs in that code.

    python -m figures.ed5      # from reproducibility/, a few seconds; needs citeseq's results
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from citeseq import plots as cs
from figures.layout import KEY_W, PLOT, ED_WIDTH as WIDTH, assemble, axes_at, save, use
from paths import require_input

GATE_LEFT, GATE_KEY, GATE_TOP = 0.7, 0.8, 0.4   # a-c: room for the title's overhang, the key, the letter above the title
TITLE = 0.25                      # space above axes that carry a title
SCATTER_H, SCATTER_GAP = 1.75, 0.12   # d, e: height, and the gap between axes that share y


def gating_panels():
    """a-c: each marker's CLR density by gate, with its key."""
    w = WIDTH / 3
    out = []
    for marker, sub in cs.gating_subsets():
        h = PLOT['bottom'] + 1.2 + GATE_TOP
        fig = plt.figure(figsize=(w, h))
        cs.draw_gating(axes_at(fig, GATE_LEFT, PLOT['bottom'], w - GATE_LEFT - GATE_KEY, 1.2), marker, sub)
        plot_style.legend_outside(fig)
        out.append(save(fig, f'ed5_{"abc"[len(out)]}_gating_{marker.lower()}'))
    return out


def scatter_row(true_lfc, panels, error, stem):
    """d or e: one axes per estimator, sharing both axes, y labelled on the first only, as lfc_*_all_methods."""
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


if __name__ == '__main__':
    use()
    res = pd.read_csv(require_input(cs.RESULTS / 'CITE_seq_results.csv', what='per-gene LFCs of every replicate',
                                    source='python -m citeseq.exp'), index_col=0)
    seurat = pd.read_csv(require_input(cs.RESULTS / 'seurat_lfc_results.csv', what="Seurat's LFCs",
                                       source='cd citeseq/R && Rscript pbmc10k_seurat.R'))
    a, b, c = gating_panels()
    true_lfc, panels = cs.all_methods(res, seurat)
    d = scatter_row(true_lfc, panels, False, 'ed5_d_lfc_vs_true_all_methods')
    e = scatter_row(true_lfc, panels, True, 'ed5_e_lfc_error_all_methods')
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)],
              [('d', d, 0)], [('e', e, 0)]], 'ed5', width=WIDTH)
