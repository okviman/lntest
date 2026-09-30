"""Fig 2: the Visium HD lymph node and PBMC10k, a block each, assembled from the arms' existing figures.

Three columns, four rows, within 170 mm. Lymph node block: a, the experiment's diagram (two columns); b-d the
pooled precision, recall and AUPRC against the pseudo-bulk reference (lymphnode.subsampling), the method key beside d. PBMC10k block, the clustering
arm's second dataset (clustering/config_pbmc10k.yaml; PBMC3k is in Extended Data): e, the diagram; f n_sig_genes,
g avg_lfc (log scale), h the false-positive rate on held-out negatives against the variance ratio, Bonferroni
(clustering.heldout_fp; BH in Extended Data Fig 9c); i the Jaccard heatmaps for all significant genes at 10 clusters, one per method, on the
colour scale they share (clustering.plots; every resolution is in Extended Data). Each plot is drawn by its own drawing code from the saved results, at the size it
prints at; a change to a plot belongs in that code.

The diagrams are drawn elsewhere and read as PDFs from `Claude outputs/` (fig2-drafts/fig2_diagrams.py).

    python -m figures.fig2      # from reproducibility/, a few seconds; needs lymphnode's and clustering's results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from clustering import heldout_fp as hf
from clustering import plots as cp
from figures.layout import LETTER, PLOT, TEXT, WIDTH, assemble, axes_at, save, use
from lymphnode import subsampling as ln
from method_colors import method_color, method_label, method_order
from paths import require_input, results_dir

LYMPH = results_dir(ln.__file__)
CLUSTERING = results_dir(cp.__file__)
PBMC = CLUSTERING / 'pbmc10k'     # the PBMC block's dataset
OUTSIDE = Path(__file__).resolve().parents[3] / 'Claude outputs'  # the two diagrams
COL = WIDTH / 3
ROWS = (1.42, 1.25, 1.91, 1.9)     # a-b, c-d, e-g, h-i: each tall enough for its labels, within 170 mm together
TOPS = (0.13, 0.13, 0.25, 0.25)    # room above each row's axes; the PBMC labels run past their axes
HEAT = dict(left=0.3, gap=0.25, cbar=0.66, top=0.35, bottom=0.84)  # i's margins, in inches
HEAT_LABEL = 5                     # i's cell types, in points: Nature's smallest text
GAP = 0.06                         # between rows
HEAT_CLUSTERS = '10'


def panel(draw, stem, row):
    """One plot in a one-column slot of this row, drawn by draw(ax); returns its file and the labels its key had."""
    h = ROWS[row]
    fig = plt.figure(figsize=(COL, h))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], COL - PLOT['left'] - PLOT['right'], h - PLOT['bottom'] - TOPS[row])
    draw(ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    if ax.get_legend():
        ax.get_legend().remove()  # one key for the figure
    return save(fig, stem), labels


def key_panel(labels, stem, row):
    """The one method key, in a slot of its own."""
    h = ROWS[row]
    fig = plt.figure(figsize=(COL, h))
    methods = method_order(labels)
    fig.legend([Patch(facecolor=method_color(m)) for m in methods], methods, title='Method', loc='upper left',
               bbox_to_anchor=(PLOT['left'] / COL, 1 - TOPS[row] / h), alignment='left')
    return save(fig, stem)


def heatmap_panel(metrics, keys, cluster_count, stem, row):
    """i: each method's Jaccard heatmap for all significant genes at HEAT_CLUSTERS clusters, without the cluster
    numbers, on one colour scale over the three."""
    groups = cp.heatmap_groups(str(metrics), 'jaccard_matrix', keys)
    rk = next(r for r in keys if cluster_count(r) == HEAT_CLUSTERS)
    vmax = cp.shared_vmax(groups, 'allsig', rk)
    methods = method_order([method_label(m) for m in cp.KNOWN_METHODS if ('allsig', m) in groups])
    by_label = {method_label(m): m for m in cp.KNOWN_METHODS}
    w, h = 2 * COL, ROWS[row]
    n = len(methods)
    hw = (w - HEAT['left'] - (n - 1) * HEAT['gap'] - HEAT['cbar']) / n
    hh = h - HEAT['top'] - HEAT['bottom']
    fig = plt.figure(figsize=(w, h))
    for k, label in enumerate(methods):
        _, mats = groups[('allsig', by_label[label])]
        cp.draw_heatmap(axes_at(fig, HEAT['left'] + k * (hw + HEAT['gap']), HEAT['bottom'], hw, hh), mats[rk], vmax,
                        label, cluster_numbers=False, label_size=HEAT_LABEL)
    fig.text((HEAT['left'] + (n * hw + (n - 1) * HEAT['gap']) / 2) / w, 1 - 0.03 / h, f'{HEAT_CLUSTERS} clusters',
             ha='center', va='top', fontsize=TEXT)
    cax = axes_at(fig, w - HEAT['cbar'] + 0.06, HEAT['bottom'], 0.08, hh)
    fig.colorbar(ScalarMappable(cmap='viridis', norm=Normalize(0, vmax)), cax=cax, label='Jaccard Index')
    return save(fig, stem)


if __name__ == '__main__':
    use()
    raw = pd.read_csv(require_input(LYMPH / 'subsampling_results_raw.csv', what='the lymph node subsampling results',
                                    source='python -m lymphnode.subsampling (see README)'))
    config = yaml.safe_load(open(require_input(CLUSTERING.parent / 'config_pbmc10k.yaml', what='the PBMC10k clustering config',
                                               source='the clustering arm')))
    metrics = PBMC / 'metrics'
    keys, order, _, cluster_count = cp.resolution_axis(str(metrics), config)
    diagrams = {name: require_input(OUTSIDE / f'fig2_{name}_diagram.pdf', what=f'the {name} diagram',
                                    source='Claude outputs/fig2-drafts/fig2_diagrams.py')
                for name in ('lymphnode', 'pbmc10k')}

    heldout = pd.read_csv(require_input(metrics / hf.CSV, what='the PBMC10k held-out false positives',
                                        source='python -m clustering.heldout_fp (see README)'))

    def box(csv, ycol, title=None, log=False):
        df = pd.read_csv(require_input(metrics / csv, what=f'the PBMC10k {csv}', source='python -m clustering.metrics'))
        return lambda ax: cp.draw_box(ax, df, ycol, cp.YLABELS[ycol], title, order, cluster_count, log=log)

    plots = {
        'b': (lambda ax: ln.draw_pooled(ax, raw, 'precision', ln.YLABELS['precision']), 'fig2_b_lymphnode_precision', 0),
        'c': (lambda ax: ln.draw_pooled(ax, raw, 'recall', ln.YLABELS['recall']), 'fig2_c_lymphnode_recall', 1),
        'd': (lambda ax: ln.draw_pooled(ax, raw, 'auprc', ln.YLABELS['auprc']), 'fig2_d_lymphnode_auprc', 1),
        'f': (box('n_sig_genes_by_resolution.csv', 'n_sig_genes'), 'fig2_f_pbmc10k_n_sig_genes', 2),
        'g': (box('avg_lfc_by_resolution.csv', 'avg_abs_lfc_sig', log=True), 'fig2_g_pbmc10k_avg_lfc', 2),
        'h': (lambda ax: hf.draw_fpr(ax, heldout), 'fig2_h_pbmc10k_heldout_fpr', 3),
    }
    drawn, labels = {}, [method_label(m) for m in raw['method'].unique()]
    for letter, (draw, stem, row) in plots.items():
        drawn[letter], seen = panel(draw, stem, row)
        labels += seen
    key = key_panel(sorted(set(labels)), 'fig2_key', 1)
    i = heatmap_panel(metrics, keys, cluster_count, 'fig2_i_pbmc10k_jaccard_heatmaps', 3)
    assemble([[('a', diagrams['lymphnode'], LETTER), ('b', drawn['b'], 0)],
              [('c', drawn['c'], 0), ('d', drawn['d'], 0), ('', key, 0)],
              [('e', diagrams['pbmc10k'], LETTER), ('f', drawn['f'], 0), ('g', drawn['g'], 0)],
              [('h', drawn['h'], 0), ('i', i, 0)]], 'fig2', gap=GAP)
