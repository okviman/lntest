"""Extended Data Fig 10: PBMC3k, the clustering arm's first and sparser dataset, assembled from its existing figures.

a-d: n_sig_genes, avg_lfc (log scale) and the Jaccard to curated markers for all significant genes and for the top 20
(clustering.plots.draw_box), one key; e: the false-positive rate on held-out negatives over all clusters, and f: by
cluster size, both Bonferroni as Fig 2h and with the Wilcoxon without tie correction (clustering.heldout_fp.draw_fpr), one key
above f. Fig 2 and
Extended Data Figs 8-9 show the same for PBMC10k. Each is drawn by the arm's own drawing code from the saved results,
at the size it prints at; a change to a panel belongs in that code.

    python -m figures.ed10     # from reproducibility/, a few seconds; needs the clustering arm's PBMC3k results
"""

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Patch

from clustering import heldout_fp as hf
from clustering import plots as cp
from figures.ed9 import held_out, row
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from method_colors import method_color, method_order
from paths import require_input, results_dir

CLUSTERING = results_dir(cp.__file__)
SLOT = WIDTH / 3
ROW, TOP = 1.9, 0.25   # a row of box plots and the room above their axes, as Fig 2's PBMC rows


def panel(draw, stem):
    """One plot in a third of the page; returns its file and the labels its key had, the key removed."""
    fig = plt.figure(figsize=(SLOT, ROW))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], SLOT - PLOT['left'] - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    draw(ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    if ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem), labels


def key_panel(labels, stem):
    """The box plots' one method key, in a slot of its own."""
    fig = plt.figure(figsize=(SLOT, ROW))
    methods = method_order(labels)
    fig.legend([Patch(facecolor=method_color(m)) for m in methods], methods, title='Method', loc='upper left',
               bbox_to_anchor=(PLOT['left'] / SLOT, 1 - TOP / ROW), alignment='left')
    return save(fig, stem)


if __name__ == '__main__':
    use()
    config = yaml.safe_load(open(require_input(CLUSTERING.parent / 'config.yaml', what='the PBMC3k clustering config',
                                               source='the clustering arm')))
    metrics = CLUSTERING / 'metrics'
    keys, order, _, cluster_count = cp.resolution_axis(str(metrics), config)
    top = config['top_genes']

    def box(csv, ycol, title=None, log=False):
        df = pd.read_csv(require_input(metrics / csv, what=f'the PBMC3k {csv}', source='python -m clustering.metrics'))
        return lambda ax: cp.draw_box(ax, df, ycol, cp.YLABELS[ycol], title, order, cluster_count, log=log)

    boxes = {
        'a': (box('n_sig_genes_by_resolution.csv', 'n_sig_genes'), 'ed10_a_pbmc3k_n_sig_genes'),
        'b': (box('avg_lfc_by_resolution.csv', 'avg_abs_lfc_sig', log=True), 'ed10_b_pbmc3k_avg_lfc'),
        'c': (box('avg_jaccard_allsig_by_resolution.csv', 'avg_jaccard_sig', cp.jaccard_title('allsig', top)),
              'ed10_c_pbmc3k_jaccard_allsig'),
        'd': (box(f'avg_jaccard_top{top}_by_resolution.csv', 'avg_jaccard_sig', cp.jaccard_title('top', top)),
              f'ed10_d_pbmc3k_jaccard_top{top}'),
    }
    drawn, labels = {}, []
    for letter, (draw, stem) in boxes.items():
        drawn[letter], seen = panel(draw, stem)
        labels += seen
    key = key_panel(sorted(set(labels)), 'ed10_key')
    df = held_out(CLUSTERING, 'PBMC3k')

    def pooled(ax):
        hf.draw_fpr(ax, df, hf.METHODS)
        ax.set_title('All clusters')

    e, _ = panel(pooled, 'ed10_e_pbmc3k_heldout_fpr')
    f = row(df, [{'size': s} for s in hf.SIZES], list(hf.SIZES), 'ed10_f_pbmc3k_heldout_fpr_by_size', key=True)
    assemble([[('a', drawn['a'], 0), ('b', drawn['b'], 0), ('', key, 0)],
              [('c', drawn['c'], 0), ('d', drawn['d'], 0), ('e', e, 0)],
              [('f', f, 0)]], 'ed10', width=WIDTH)
