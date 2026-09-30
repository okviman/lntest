"""Extended Data Fig 7: the Visium HD lymph node, assembled from the arm's existing figures.

a: the DE time per test against p_sub, both compartments pooled (lymphnode.subsampling), with its method key.
Drawn by the arm's own drawing code from the saved results, at the size it prints at; a change to it belongs in
that code.

    python -m figures.ed7      # from reproducibility/, a few seconds; needs lymphnode's results
"""

import matplotlib.pyplot as plt
import pandas as pd

from figures.layout import KEY_W, PLOT, ROW_H, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from lymphnode import subsampling as ln
from paths import require_input, results_dir

LYMPH = results_dir(ln.__file__)


def runtime(raw, stem):
    """The runtime plot in half the page, wide enough for its p_sub labels, its key beside it."""
    slot = (WIDTH - KEY_W) / 2
    fig = plt.figure(figsize=(slot + KEY_W, ROW_H))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], slot - PLOT['left'] - PLOT['right'],
                 ROW_H - PLOT['bottom'] - PLOT['top'])
    df = ln.runtime_table(raw)
    ln.draw_pooled(ax, df, 'de_time_s', ln.YLABELS['de_time_s'])
    ln.method_legend(fig, df)
    return save(fig, stem)


if __name__ == '__main__':
    use()
    raw = pd.read_csv(require_input(LYMPH / 'subsampling_results_raw.csv', what='the lymph node subsampling results',
                                    source='python -m lymphnode.subsampling (see README)'))
    assemble([[('a', runtime(raw, 'ed7_a_lymphnode_runtime'), 0)]], 'ed7', width=WIDTH)
