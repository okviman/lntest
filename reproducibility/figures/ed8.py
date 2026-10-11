"""Extended Data Fig 8: the PBMC clusters' DEGs and their effect sizes, assembled from clustering.effect_sizes' drawing
code.

Every Leiden cluster at five resolutions against the rest. a, b, PBMC10k and PBMC3k: each test's significant genes
(Bonferroni 0.05) per cluster; c, d: their mean |LFC| as each package reports it (LN's estimate, Scanpy's for the log1p
t-test, Seurat 5's for the Wilcoxon), with the difference of log1p means the log1p t-test tests, lighter. A box per
series at each resolution, every cluster a point, each box under its counterpart above; one log axis per row, a key per
row. Each plot is drawn by its own drawing code from the saved results, at the size it prints at; a change to a plot
belongs in that code.

    python -m figures.ed8      # from reproducibility/, seconds; needs clustering.effect_sizes' results
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from clustering import effect_sizes as es
from figures.layout import KEY_W, PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from paths import require_input

ROW, TOP, LEFT = 2.1, 0.25, 0.62   # a row, the room above its axes for a title, left of them for the y label
SLOT = (WIDTH - KEY_W) / 2


def panel(draw, title, stem):
    """One plot in half the page less the key, titled if title."""
    fig = plt.figure(figsize=(SLOT, ROW))
    top = TOP if title else PLOT['top']
    draw(axes_at(fig, LEFT, PLOT['bottom'], SLOT - LEFT - PLOT['right'], ROW - PLOT['bottom'] - top))
    return save(fig, stem)


def key(handles, labels, stem, top):
    """A key of filled boxes in a slot of its own, level with its row's axes."""
    fig = plt.figure(figsize=(KEY_W, ROW))
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1 - top / ROW), frameon=False,
               fontsize=plot_style.SMALL, handlelength=0.8, handletextpad=0.4, borderaxespad=0, labelspacing=0.8)
    return save(fig, stem)


if __name__ == '__main__':
    use()
    sizes = {name: pd.read_csv(require_input(csv, what=f'the {name} effect sizes', source='python -m clustering.effect_sizes'),
                               dtype={'cluster': str}) for name, csv in es.CSVS.items()}
    degs_y, lfc_y = es.value_range(sizes.values(), 'n_degs'), es.value_range(sizes.values())
    a, b = (panel(lambda ax, n=name: (es.draw_degs(ax, sizes[n], degs_y), ax.set_xlabel(''), ax.set_title(n)), True,
                  f'ed8_{letter}_{name.lower()}_degs') for letter, name in (('a', 'PBMC10k'), ('b', 'PBMC3k')))
    c, d = (panel(lambda ax, n=name: es.draw_effect_sizes(ax, sizes[n], lfc_y), False,
                  f'ed8_{letter}_{name.lower()}_effect_sizes') for letter, name in (('c', 'PBMC10k'), ('d', 'PBMC3k')))
    methods = sizes['PBMC10k']['method'].unique()
    assemble([[('a', a, 0), ('b', b, 0), ('', key(*es.count_key(methods), 'ed8_key_tests', TOP), 0)],
              [('c', c, 0), ('d', d, 0), ('', key(*es.key(methods, wrap=True), 'ed8_key_effect_sizes', PLOT['top']), 0)]],
             'ed8', width=WIDTH)
