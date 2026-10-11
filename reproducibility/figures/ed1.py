"""Extended Data Fig 1: the synthetic NB result beyond the null, assembled from the arm's existing figures.

a, b: lfc_ci_ln and lfc_ci_log1p (synthetic_nb.lfc_confidence_intervals).
c, d: degs_metrics_sparse and degs_metrics_dense (synthetic_nb.de_ratio), each heading centred over its axes.
Each is drawn by its own drawing code, at the size it prints at; a change to a panel belongs in that code.

    python -m figures.ed1      # from reproducibility/, a few seconds; needs synthetic_nb's planted-DEG runs
"""

import matplotlib.pyplot as plt

from figures.layout import KEY_W, PLOT, TEXT, ED_WIDTH as WIDTH, assemble, axes_at, save, use
from synthetic_nb import de_ratio, lfc_confidence_intervals as lfc_ci

LFC_H = 2.3      # a and b
ROW_H = 2.1      # c and d
LEFT, GAP = 0.55, 0.12   # c and d: left of the first axes, and between axes that share y
HEADING, TITLE = 0.2, 0.2


def lfc_panels():
    """a and b: lfc_ci_ln and lfc_ci_log1p."""
    sim = lfc_ci.simulate()
    w = WIDTH / 2
    out = []
    for key, stem in (('LN', 'ed1_a_lfc_ci_ln'), ('log1p', 'ed1_b_lfc_ci_log1p')):
        fig = plt.figure(figsize=(w, LFC_H))
        ax = axes_at(fig, PLOT['left'], PLOT['bottom'], w - PLOT['left'] - 0.15, LFC_H - PLOT['bottom'] - 0.25)
        lfc_ci.draw(ax, sim, key)
        out.append(save(fig, stem))
    return out


def metric_row(setting, stem):
    """c or d: degs_metrics for one setting, one axes per metric sharing y, its method key beside them."""
    df = de_ratio.load(setting)
    n = len(de_ratio.METRICS)
    aw = (WIDTH - KEY_W - LEFT - (n - 1) * GAP - 0.05) / n
    bottom, top = PLOT['bottom'], PLOT['top'] + HEADING + TITLE
    ah = ROW_H - bottom - top
    fig = plt.figure(figsize=(WIDTH, ROW_H))
    axes = []
    for k, (col, name) in enumerate(de_ratio.METRICS.items()):
        ax = axes_at(fig, LEFT + k * (aw + GAP), bottom, aw, ah, sharey=axes[0] if axes else None)
        de_ratio.draw_metric(ax, df, col)
        ax.set_title(name)
        if axes:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    axes[0].set_ylabel('Score')
    right = LEFT + n * aw + (n - 1) * GAP
    centre = (LEFT + right) / 2 / WIDTH
    fig.text(centre, 0.03 / ROW_H, de_ratio.XLABEL, ha='center', va='bottom', fontsize=TEXT)
    fig.text(centre, 1 - PLOT['top'] / ROW_H, de_ratio.TITLES[setting], ha='center', va='top', fontsize=TEXT)
    fig.legend(handles=de_ratio.method_handles(df.method), frameon=False, loc='center left',
               bbox_to_anchor=((right + 0.05) / WIDTH, (bottom + ah / 2) / ROW_H), alignment='left')
    return save(fig, stem)


if __name__ == '__main__':
    use()
    a, b = lfc_panels()
    c = metric_row('sparse', 'ed1_c_degs_sparse')
    d = metric_row('dense', 'ed1_d_degs_dense')
    assemble([[('a', a, 0), ('b', b, 0)],
              [('c', c, 0)],
              [('d', d, 0)]], 'ed1', width=WIDTH)
