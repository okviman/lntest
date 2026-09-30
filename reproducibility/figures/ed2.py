"""Extended Data Fig 2: the kidney design, assembled from the arm's existing figures.

a, b: capsule_qc's two panels (kidney.capsule_qc): the capsule sizes split at the median, and the DEGs each
method calls at full depth as the smallest capsules are removed, with its method key.
Each is drawn by its own drawing code from the saved results, at the size it prints at; a change to a panel
belongs in that code.

    python -m figures.ed2      # from reproducibility/, a few seconds; needs kidney's results and data
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from figures.layout import KEY_W, PLOT, ROW_H, ED_WIDTH as WIDTH, assemble, axes_at, save, use
from kidney import capsule_qc, full_depth
from paths import require_input, results_dir

KIDNEY = results_dir(full_depth.__file__)
SLOT = (WIDTH - KEY_W) / 2      # a, and b without its key


def capsule_panels():
    """a and b: capsule_qc's size histogram, and its removal scan with the method key."""
    umis = full_depth.load()
    scan = pd.read_csv(require_input(KIDNEY / 'capsule_removal_scan.csv', what='the capsule removal scan',
                                     source='python -m kidney.capsule_qc'))
    aw, ah = SLOT - PLOT['left'] - PLOT['right'], ROW_H - PLOT['bottom'] - PLOT['top']

    fig = plt.figure(figsize=(SLOT, ROW_H))
    capsule_qc.draw_sizes(axes_at(fig, PLOT['left'], PLOT['bottom'], aw, ah), umis)
    a = save(fig, 'ed2_a_capsule_sizes')

    fig = plt.figure(figsize=(SLOT + KEY_W, ROW_H))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], aw, ah)
    capsule_qc.draw_scan(ax, scan)
    plot_style.legend_outside(fig, *ax.get_legend_handles_labels())
    return a, save(fig, 'ed2_b_removal_scan')


if __name__ == '__main__':
    use()
    a, b = capsule_panels()
    assemble([[('a', a, 0), ('b', b, 0)]], 'ed2', width=WIDTH)
