"""Extended Data Fig 9: PBMC10k's false positives on held-out negatives in full, assembled from the clustering arm's results.

a: the false-positive rate against the variance ratio by cluster size (under 100, 100-400, over 400 cells); b: by the
tolerance that makes a gene a negative (means within 3.5%, 7.2%, 14.9%); c: by multiple-testing correction, Bonferroni as
in Fig 2h, a and b, BH, and none (p < 0.05); all with the Wilcoxon without tie correction (scanpy's current default)
beside Fig 2h's three tests
(clustering.heldout_fp.draw_fpr). A row's panels share their y-axis and x label, each labelled once;
one method key, in a line above a, leaves the panels wide enough for their tick labels. Drawn by the arm's own drawing code from the saved results, at the size it prints at; a change to
a panel belongs in that code.

    python -m figures.ed9      # from reproducibility/, a few seconds; needs python -m clustering.heldout_fp's CSV
"""

import matplotlib.pyplot as plt
import pandas as pd

from clustering import heldout_fp as hf
from figures.layout import PLOT, ROW_H, assemble, axes_at, method_key, save, use
from figures.layout import ED_WIDTH as WIDTH
from paths import require_input, results_dir

INNER = 0.1    # left margin of a panel sharing its row's y-axis
KEY_H = 0.35   # the key's line above the first row
TOP = 0.3      # room for the panel titles


def held_out(results, dataset):
    """A dataset's held-out false positives, from its clustering results directory."""
    return pd.read_csv(require_input(results / 'metrics' / hf.CSV, what=f'the {dataset} held-out false positives',
                                     source='python -m clustering.heldout_fp (see README)'))


def row(df, panels, titles, stem, key=False, ylabel=None):
    """draw_fpr panels, each drawn with summary()'s keywords in panels, on one y range; the key above them if key;
    ylabel in place of the first panel's own."""
    zero, top = hf.shared_range(df, panels, hf.METHODS)
    n = len(panels)
    aw = (WIDTH - PLOT['left'] - (n - 1) * INNER - n * PLOT['right']) / n
    height = ROW_H + (KEY_H if key else 0)
    fig = plt.figure(figsize=(WIDTH, height))
    x = PLOT['left']
    for k, (kw, title) in enumerate(zip(panels, titles)):
        ax = axes_at(fig, x, PLOT['bottom'], aw, ROW_H - PLOT['bottom'] - TOP)
        hf.draw_fpr(ax, df, hf.METHODS, zero=zero, top=top, **kw)
        ax.set_title(title)
        if k:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        elif ylabel:
            ax.set_ylabel(ylabel)
        if k != n // 2:
            ax.set_xlabel('')
        x += aw + PLOT['right'] + INNER
    if key:
        method_key(fig, hf.METHODS, 0.5, 1, ncol=len(hf.METHODS), loc='upper center')
    return save(fig, stem)


if __name__ == '__main__':
    use()
    df = held_out(results_dir(hf.__file__) / 'pbmc10k', 'PBMC10k')
    a = row(df, [{'size': s} for s in hf.SIZES], list(hf.SIZES), 'ed9_a_pbmc10k_heldout_fpr_by_size', key=True)
    b = row(df, [{'tolerance': t} for t in hf.TOLERANCES], [hf.tolerance_title(t) for t in hf.TOLERANCES],
            'ed9_b_pbmc10k_heldout_fpr_by_tolerance')
    c = row(df, [{'correction': c} for c in hf.CORRECTIONS], [hf.correction_title(c) for c in hf.CORRECTIONS],
            'ed9_c_pbmc10k_heldout_fpr_by_correction', ylabel='FPR')
    assemble([[('a', a, 0)], [('b', b, 0)], [('c', c, 0)]], 'ed9', width=WIDTH)
