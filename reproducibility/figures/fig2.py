"""Fig 2: the Visium HD lymph node, PBMC10k's multi-resolution clustering and the Abdelfattah et al. 2022 rerun, a block
each, assembled from the arms' drawing code.

Lymph node (lymphnode.depth_imbalance and depth_gsea, the spot version: the T-cell zone's 2 um spots kept with
probability p_sub, the B-cell follicles' all): a, the experiment's diagram; b, each test's AUPRC of the T-cell zone's
pseudo-bulk markers against p_sub; c, the Spearman correlation of each test's Hallmark NES with the pseudo-bulk
reference's (NES agreement); b and c the mean over the replicates with bars of one SD, as Fig 1's error bars. PBMC10k:
d, the diagram; e, every cluster at five Leiden resolutions, its variance ratio to the rest at equal mean (2 to the
median |log2 variance ratio| over the genes whose means agree within 7%, from 1 as Fig 1's Var(Y) / Var(X)), a box at
each number of clusters, every cluster a point (clustering.multires_gaps); f, the share of each rule's top 20 that are
housekeeping genes and the share of the cluster's protein markers in it, the mean over each resolution's clusters with
bars of one standard error, side by side (clustering.marker_rule; LN's markers and LN's score in its colour); e and f
space the resolutions evenly. Abdelfattah et al. 2022 (published.abdelfattah2022): g, the NES heatmaps of their Fig. 4d
pathways, their GSEA run as the authors ran it on the Wilcoxon's z-score (scanpy's score) and on LN's score, in square
cells on one colour range, the sets by short names; h, per cluster, how many of those pathways keep their call (FDR <
0.05) under LN, lose it, reverse it or gain one; i, MC3 over MC4 in the pathways behind the interpretation the rerun
contradicts, on one x axis, the Wilcoxon's and LN's NES side by side. The authors' own ranking, the Wilcoxon AUC, is
Extended Data Fig 10.
A method key right of c, a key of the marker rules right of f. Each plot is drawn by its own drawing code from the saved results, at the size it prints at; a change to a
plot belongs in that code.

Panel a's images are drawn elsewhere and read as a PDF from `Claude outputs/` (fig2-drafts/fig2_diagrams.py).

    python -m figures.fig2      # from reproducibility/, a few seconds; needs the lymph node, clustering and published results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import plot_style
from clustering import lfc_se as ls
from clustering import marker_rule as mk
from clustering import multires_gaps as mg
from figures import ed_abdelfattah as ea
from figures.layout import LETTER, LETTER_PT, PLOT, WIDTH, assemble, axes_at, pdf_size, save, use
from lymphnode import depth_gsea as dg
from lymphnode import depth_imbalance as di
from method_colors import method_color, method_label
from paths import require_input
from published import abdelfattah2022 as ab

THIN = 'bins'                      # the lymph node version in the figure; 'reads' for the read-thinning one
OUTSIDE = Path(__file__).resolve().parents[3] / 'Claude outputs'  # the lymph node's images
ROW_A, ROW_B = 1.42, 2.1           # the lymph node row (its images' height) and the PBMC row
F_W = 3.05                         # f's width; d and e take the rest of the row
HEAT_COL, CBAR_LABELS = 0.1, 0.42  # g's heatmap columns and the room for its colour bar's labels, narrowed for i
COMP, HEAT_VMAX = 'wilcoxon_z', 3   # the Wilcoxon as scanpy scores it, and g's colour range for its and LN's NES
KEY_W = 0.85                       # a key's slot, right of c and of f
F = (('housekeeping', 'Fraction of top 20'), ('protein_recall', 'Recall in top 20'))   # f's plots, side by side
TOP = 0.25                         # room above a plot's axes for its title
GAP = 0.06                         # between rows


def read(path, what, source, **kw):
    return pd.read_csv(require_input(path, what=what, source=source), **kw)


def panel(draw, title, stem, width, height, left=PLOT['left']):
    """One plot in a slot of the given size, titled if title; any legend it drew is dropped for the figure's key."""
    fig = plt.figure(figsize=(width, height))
    top = TOP if title else PLOT['top']
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], height - PLOT['bottom'] - top)
    draw(ax)
    if title:
        ax.set_title(title)
    if ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


def beside(draws, stem, width, height, left=PLOT['left'], gap=0.45, ylabel=None, xlabel=None):
    """Plots side by side in one slot, each with its own y scale and title; ylabel and xlabel, if given, one each for
    all of them."""
    fig = plt.figure(figsize=(width, height))
    n = len(draws)
    w = (width - left - PLOT['right'] - gap * (n - 1)) / n
    for k, draw in enumerate(draws):
        ax = axes_at(fig, left + k * (w + gap), PLOT['bottom'], w, height - PLOT['bottom'] - TOP)
        draw(ax)
        if xlabel:
            ax.set_xlabel('')
    if ylabel:
        fig.text(0.12 / width, (PLOT['bottom'] + (height - TOP)) / 2 / height, ylabel, rotation=90, ha='center',
                 va='center')
    if xlabel:
        fig.text((left + width - PLOT['right']) / 2 / width, 0.08 / height, xlabel, ha='center', va='bottom')
    return save(fig, stem)


def umaps_over_boxes(umap, clusters, gaps, width, height, stem):
    """d, the UMAP coloured by its clusters at each resolution, side by side under an arrow of increasing resolution; e,
    below, the variance ratio at each resolution, each box under its UMAP; with the letter strip, e's letter drawn
    here."""
    fig = plt.figure(figsize=(width, height))
    left, right = LETTER + 0.45, PLOT['right']
    cw = (width - left - right) / clusters.shape[1]                  # one column per resolution
    span = umap.max(0) - umap.min(0)
    uw = cw - 0.06
    uh = uw * span[1] / span[0]
    band = 0.27                                                       # the arrow and its label, above the UMAPs
    bottom = height - band - uh
    for k, key in enumerate(clusters.columns):
        mg.draw_clusters(axes_at(fig, left + k * cw + 0.03, bottom, uw, uh), umap, clusters[key])
    y = (height - 0.21) / height
    fig.add_artist(FancyArrowPatch(((left + 0.03) / width, y), ((width - right - 0.03) / width, y),
                                   transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=0.8, color='0.3'))
    fig.text((left + width - right) / 2 / width, 1 - 0.02 / height, 'Clustering resolution', ha='center', va='top')
    top = bottom - 0.14
    mg.draw_gaps(axes_at(fig, left, PLOT['bottom'], width - left - right, top - PLOT['bottom']), gaps, 'PBMC10k',
                 'gap_equal_mean')
    fig.text(0, (top + 0.1) / height, 'e', fontsize=LETTER_PT, fontweight='bold', ha='left', va='top')
    return save(fig, stem)


def key(entries, height, stem):
    """A key in a slot of its own, its entries from the top, as Fig 1's: (colour, alpha, label) each."""
    fig = plt.figure(figsize=(KEY_W, height))
    line = dict(marker='o', ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'])
    labels = [l.replace(' (score rank)', '\n(score rank)') for _, _, l in entries]   # fits the slot
    fig.legend([Line2D([], [], color=c, alpha=a, **line) for c, a, _ in entries], labels,
               loc='upper left', bbox_to_anchor=(0, 1 - TOP / height), frameon=False, fontsize=plot_style.SMALL,
               handlelength=1.4, borderaxespad=0)
    return save(fig, stem)


if __name__ == '__main__':
    use()
    images = require_input(OUTSIDE / 'fig2_lymphnode_diagram.pdf', what="the lymph node's images",
                           source='Claude outputs/fig2-drafts/fig2_diagrams.py')
    umap, clusters = mg.umap_clusters('PBMC10k')
    source = f'python -m lymphnode.{{}} --config lymphnode/config_50rep.yaml --thin {THIN}'
    results = read(di.OUTS[THIN] / 'results.csv', 'the depth-imbalance results', source.format('depth_imbalance'))
    gsea = read(dg.OUTS[THIN] / 'agreement.csv', 'the depth-GSEA agreement', source.format('depth_gsea'))
    gaps = read(mg.OUT / 'comparisons.csv', 'the multires_gaps comparisons', 'python -m clustering.multires_gaps')
    per = read(mk.OUT / 'per_list.csv', 'the marker_rule per-list table', 'python -m clustering.marker_rule')
    t = ab.tables()

    # the lymph node: the diagram, then two plots in the rest of the row
    w = (WIDTH - LETTER - pdf_size(images)[0] - KEY_W) / 2
    b = panel(lambda ax: di.draw_errorbar(ax, results[results['cluster'] == di.T_ZONE], 'auprc',
                                     'DGE AUPRC', di.XLABEL[THIN]),
              None, 'fig2_b_lymphnode_auprc', w, ROW_A)
    c = panel(lambda ax: dg.draw_agreement(ax, gsea, thin=THIN, ylabel='GSEA NES\ncorrelation'), None,
              'fig2_c_lymphnode_gsea', w, ROW_A)

    # PBMC10k: the clusters at each resolution over their variance ratios, the top 20s against resolution
    de = umaps_over_boxes(umap, clusters, gaps, WIDTH - F_W - KEY_W, ROW_B, 'fig2_de_pbmc10k_clusters_gap')
    f_draws = []
    for metric, ylabel in F:
        def draw(ax, metric=metric, ylabel=ylabel):
            mk.draw_by_resolution(ax, per, 'PBMC10k', metric, ylabel=ylabel)
            ax.set_title(mk.TITLES[metric])
        f_draws.append(draw)
    f = beside(f_draws, 'fig2_f_pbmc10k_markers', F_W, ROW_B, left=0.5, gap=0.55, xlabel='Number of clusters')

    # Abdelfattah: the heatmaps, the calls that change per cluster, and MC3 against MC4
    g, height = ea.heatmaps(t, COMP, 'fig2', letter='g', shared_xlabel=True, col=HEAT_COL, cbar_labels=CBAR_LABELS,
                            vmax=HEAT_VMAX)
    h = ea.call_changes_panel(t['gsea'], height, 'fig2_h_abdelfattah_call_changes', comp=COMP)
    i = ea.contrast_by_cluster(t, WIDTH - sum(pdf_size(x)[0] for x in (*g, h)) - 0.01, height, 'fig2_i_abdelfattah_contrast',
                               comp=COMP)

    methods = [(method_color(m), 1.0, method_label(m)) for m in ('ln', 't_test', 'wilcoxon')]
    rules = [(*ls.MARKER_STYLE[r], ls.MARKER_RULES[r]) for r in mk.RULES]
    assemble([[('a', images, LETTER), ('b', b, 0), ('c', c, 0), ('', key(methods, ROW_A, 'fig2_key_methods'), 0)],
              [('d', de, 0), ('f', f, 0), ('', key(rules, ROW_B, 'fig2_key_rules'), 0)],
              [('g', g[0], 0), *[('', x, 0) for x in g[1:]], ('h', h, 0), ('i', i, 0)]], 'fig2', gap=GAP)
