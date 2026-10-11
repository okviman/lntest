"""Fig 1: panels c-h drawn at the size they print at, and the whole figure assembled from them and a, b.

The figure is 183 mm (7.2 in) wide, Nature's double column. Every panel is placed at 100%, never scaled,
so its text prints at the size it was drawn: 7 pt for axis labels, ticks and titles, as in panels a and
b, and plot_style's 6 pt for legends. Every axes is placed in inches, so f, g and h have the same axes.
The method colours are keyed once, beside h, and named by panel a's headings; c draws all three tests.

Panels a and b are drawn elsewhere and read as PDFs from `Claude outputs/` beside the repositories.
Each row's panels are placed left to right and top-aligned, and the rows top to bottom; a row wider
than the figure stops the script instead of overlapping.

    python -m figures.fig1      # from reproducibility/, a few seconds; needs the arms' results

Reads synthetic_nb/results (null_ratio's sweeps), kidney/data (the capsule figures) and
kidney/results (the spot-splitting runs), and writes figures/results/panels/fig1_*.pdf and figures/results/paper/fig1.pdf.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.kidney_panels import null_auc_curves
from figures.layout import LETTER, WIDTH, assemble, axes_at, pdf_size, save, use
from kidney.capsule_figure import BAND, capsule_crop, draw_overview, draw_splits, section, split_legend
from kidney import fpr_plots
from kidney.fpr_plots import SPLIT
from paths import require_input, results_dir
from synthetic_nb import null_ratio

KIDNEY = results_dir(fpr_plots.__file__)
OUTSIDE = Path(__file__).resolve().parents[3] / 'Claude outputs'  # panels a and b
MUS = (5, 50)
SEED = 0
DE_SCALE = 0.9   # d and e's share of their row, so that the figure fits 170 mm


def panel_c(slot):
    """The synthetic NB null: each test's FPR against Var(Y)/Var(X), at base mu 5 and 50."""
    df = null_ratio.load(MUS)
    multiples = null_ratio.var_multiples(df[df.base_mu == MUS[0]], MUS[0])
    w, h = slot
    left, gap, key = 0.5, 0.15, 0.62
    aw = (w - left - gap - key) / 2
    bottom, top = 0.4, 0.25
    fig = plt.figure(figsize=slot)
    axes = [axes_at(fig, left, bottom, aw, h - bottom - top)]
    axes.append(axes_at(fig, left + aw + gap, bottom, aw, h - bottom - top, sharey=axes[0]))
    for ax, mu in zip(axes, MUS):
        null_ratio.panel(ax, df[df.base_mu == mu], mu)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel('FPR')
    axes[0].set_ylim(-0.03, 1.03)
    fig.legend(handles=null_ratio.var_handles(multiples), title=r'Var$(X)$', handlelength=3,
               loc='upper left', bbox_to_anchor=((left + 2 * aw + gap + 0.05) / w, 1 - top / h), alignment='left')
    return save(fig, 'fig1_c_null_fpr')


def panels_de():
    """The capsule overview and one capsule's splits at the same image height, DE_SCALE of the width the row allows;
    the width left over is a gap between them."""
    sec = section()
    crop, box = capsule_crop(sec)
    aspect = crop.shape[0] / crop.shape[1]
    gap, title, bottom, key = 0.05, 0.2, 0.05, 0.7
    # d is BAND times as wide as it is tall; e's three images are each 1 / aspect as wide as tall
    full = (WIDTH - 2 * LETTER - 2 * gap - key) / (1 + BAND * aspect / 3)
    images = DE_SCALE * full
    img_h = aspect * images / 3
    height = title + img_h + bottom

    d_w = BAND * img_h
    spare = (1 + BAND * aspect / 3) * (full - images)
    fig = plt.figure(figsize=(LETTER + d_w + spare, height))
    draw_overview(axes_at(fig, LETTER, bottom, d_w, img_h), sec)
    d = save(fig, 'fig1_d_capsule_overview')

    e_w = LETTER + images + 2 * gap + key
    fig = plt.figure(figsize=(e_w, height))
    axes = [axes_at(fig, LETTER + k * (images / 3 + gap), bottom, images / 3, img_h) for k in range(3)]
    draw_splits(axes, sec, crop, box, SEED)
    split_legend(fig, loc='upper left', bbox_to_anchor=((LETTER + images + 2 * gap + 0.05) / e_w, 1))
    return d, save(fig, 'fig1_e_capsule_split')


def plot_row():
    """f, g and h: the spot-splitting FPR, AUPRC and PR curves, with equal axes; the keys beside h."""
    null = require_input(KIDNEY / 'split_null_results.csv', what='the no-DEG spot-splitting results',
                         source='python -m kidney.spot_split (see README)')
    results = require_input(KIDNEY / 'split_degs_results.csv', what='the with-DEG spot-splitting results',
                            source='python -m kidney.spot_split (see README)')
    json_file = require_input(KIDNEY / 'split_degs_pr_curve_data.json', what='the with-DEG PR curves',
                              source='python -m kidney.spot_split (see README)')
    curve_data = json.load(open(json_file))
    return null_auc_curves(pd.read_csv(null), pd.read_csv(results), curve_data, SPLIT,
                           ('fig1_f_split_null_fpr', 'fig1_g_split_auprc', 'fig1_h_split_pr_curves'))


if __name__ == '__main__':
    use()
    a = require_input(OUTSIDE / 'fig1a.pdf', what='Fig 1a, the cluster-marker overview',
                      source='Claude outputs/fig1a-drafts/fig1_ab.py')
    b = require_input(OUTSIDE / 'fig1b.pdf', what='Fig 1b, the mechanism schematic',
                      source='Claude outputs/fig1a-drafts/fig1_ab.py')
    b_w, b_h = pdf_size(b)
    c = panel_c((WIDTH - LETTER - b_w, b_h))  # c takes what b leaves of its row
    d, e = panels_de()
    f, g, h = plot_row()
    assemble([[('a', a, LETTER)],
              [('b', b, LETTER), ('c', c, 0)],
              [('d', d, 0), ('e', e, 0)],
              [('f', f, 0), ('g', g, 0), ('h', h, 0)]], 'fig1')
