"""Extended Data Fig 4: DEGs of the PBMC clusters and their effect sizes, assembled from the drawing code of
clustering.effect_sizes.

Every Leiden cluster at five resolutions is tested against the rest.
a, b: the number of significant genes (Bonferroni 0.05) per cluster for each test, in PBMC10k (a) and PBMC3k (b).
c, d: their mean |LFC| as each package reports it: LN's estimate, scanpy's for the log1p t-test and Seurat 5's for the
Wilcoxon. The difference of log1p means, which the log1p t-test tests, is drawn lighter.
One box per series at each resolution, every cluster a point, each box under its counterpart above. One log axis and
one key per row.

    python -m figures.edfig4      # from reproducibility/, seconds; needs the results of clustering.effect_sizes
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
                  f'edfig4_{letter}_{name.lower()}_degs') for letter, name in (('a', 'PBMC10k'), ('b', 'PBMC3k')))
    c, d = (panel(lambda ax, n=name: es.draw_effect_sizes(ax, sizes[n], lfc_y), False,
                  f'edfig4_{letter}_{name.lower()}_effect_sizes') for letter, name in (('c', 'PBMC10k'), ('d', 'PBMC3k')))
    methods = sizes['PBMC10k']['method'].unique()
    assemble([[('a', a, 0), ('b', b, 0), ('', key(*es.count_key(methods), 'edfig4_key_tests', TOP), 0)],
              [('c', c, 0), ('d', d, 0), ('', key(*es.key(methods, wrap=True), 'edfig4_key_effect_sizes', PLOT['top']), 0)]],
             'edfig4', width=WIDTH)
