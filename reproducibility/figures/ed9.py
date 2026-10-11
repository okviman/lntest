"""Extended Data Fig 9: the Abdelfattah et al. 2022 rerun beyond Fig 2, assembled from published.abdelfattah2022's
drawing code.

a, the myeloid cells' UMAP coloured by the paper's clusters and by counts per cell; b, each cluster's depth relative to
the other myeloid cells; c, each cluster's variance ratio at equal mean against its |log2 depth ratio|, with its
least-squares line, as Fig 2e's (published.abdelfattah2022 --variance); d, each cluster's DEGs (Bonferroni 0.05) under
the Wilcoxon and LN, as b's bars; e, each test's DEGs against the cluster's number of cells, with its least-squares line
and Spearman correlation; f, the Wilcoxon's DEGs over LN's against the cluster's |log2 depth ratio|, as c
(published.abdelfattah2022 --calls); g, each cluster's expressed genes against its log2 depth ratio, the Wilcoxon's
z-score (scanpy's score) and LN's score, on one y range; h, each cluster's Hallmark NES against its log2 depth ratio, every set a point
over its box, the Wilcoxon's and LN's (MC3 against MC4 is Fig 2i; the authors' AUC is Extended Data Fig 10). The
same figure with
every panel and with the Wilcoxon's z or the log1p t-test in place of the AUC is figures.ed_abdelfattah's. Each plot is
drawn by its own drawing code from the saved results, at the size it prints at; a change to a plot belongs in that code.

    python -m figures.ed9      # from reproducibility/, seconds; needs published/results/
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from figures import ed_abdelfattah as ea
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from paths import require_input
from published import abdelfattah2022 as ab

COMP = 'wilcoxon_z'            # the Wilcoxon as scanpy scores it; the authors' AUC is Extended Data Fig 10
TAG = 'ed9'
ROW, TOP = 2.0, 0.25                # the second row and the room above its axes for a title
C_W = 1.8                           # c's width; b takes the rest of the first row


def genes(t, values, k, m, ylabel, ref, comp=COMP, stem=None, ylim=None):
    """g: one method's expressed genes against each cluster's depth, in a quarter of the page, on ylim if given (the
    first of the pair labelled)."""
    fig = plt.figure(figsize=(ea.QUARTER, ROW))
    left = 0.5
    ax = axes_at(fig, left, PLOT['bottom'], ea.QUARTER - left - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    ab.draw_by_depth(ax, values['genes', m], t['clusters'], m, ylabel, ref, ylim=ylim, show_ylabel=ylim is None or k == 0,
                     points=False)   # ~9,000 genes a box
    ax.set_title(ab.TITLE[(comp, 'ln')[k]])
    return save(fig, stem or f'{TAG}_g{k + 1}_genes_{m}')


def variance(t, width):
    """c: each cluster's variance ratio against its log2 depth ratio, beside the depth ratios."""
    var = pd.read_csv(require_input(ab.OUT / 'variance.csv', what='the clusters\' variance ratios',
                                    source='python -m published.abdelfattah2022 --variance'))
    fig = plt.figure(figsize=(width, ea.ROW1))
    left = 0.55
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], ea.ROW1 - PLOT['bottom'] - TOP)
    ab.draw_variance_vs_depth(ax, t['clusters'], var)
    return save(fig, f'{TAG}_c_variance_vs_depth')


def pathways(t, values, k, m, ylim, stem=None):
    """h: one method's Hallmark NES against each cluster's depth, every set a point over its box, in a quarter of the
    page."""
    fig = plt.figure(figsize=(ea.QUARTER, ROW))
    left = 0.5
    ax = axes_at(fig, left, PLOT['bottom'], ea.QUARTER - left - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    ab.draw_by_depth(ax, values['pathways', m], t['clusters'], m, 'NES', 0.0, ylim=ylim, show_ylabel=k == 0)
    ax.set_title(ab.TITLE[m])
    return save(fig, stem or f'{TAG}_h{k + 1}_pathways_{m}')


def calls(width):
    """d: each cluster's DEGs under the Wilcoxon and LN, as b's bars, with the key to the right."""
    table = pd.read_csv(require_input(ab.OUT / 'calls.csv', what='the calls table',
                                      source='python -m published.abdelfattah2022 --calls'))
    fig = plt.figure(figsize=(width, ea.ROW1))
    left, key = 0.6, 1.0
    ax = axes_at(fig, left, PLOT['bottom'], width - left - key, ea.ROW1 - PLOT['bottom'] - TOP)
    ab.draw_calls(ax, table)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', frameon=False, fontsize=plot_style.SMALL,
               bbox_to_anchor=((width - key + 0.08) / width, 1 - TOP / ea.ROW1), handlelength=1.0, borderaxespad=0)
    return save(fig, f'{TAG}_d_calls')


def calls_vs_cells(width):
    """e: each cluster's DEGs under each test against its number of cells."""
    table = pd.read_csv(require_input(ab.OUT / 'calls.csv', what='the calls table',
                                      source='python -m published.abdelfattah2022 --calls'))
    fig = plt.figure(figsize=(width, ea.ROW1))
    left = 0.6
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], ea.ROW1 - PLOT['bottom'] - TOP)
    ab.draw_calls_vs_cells(ax, table)
    return save(fig, f'{TAG}_e_calls_vs_cells')


def calls_vs_depth(width):
    """f: each cluster's Wilcoxon DEGs over LN's against its |log2 depth ratio|, as c."""
    table = pd.read_csv(require_input(ab.OUT / 'calls.csv', what='the calls table',
                                      source='python -m published.abdelfattah2022 --calls'))
    fig = plt.figure(figsize=(width, ea.ROW1))
    left = 0.6
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], ea.ROW1 - PLOT['bottom'] - TOP)
    ab.draw_calls_vs_depth(ax, table)
    return save(fig, f'{TAG}_f_calls_vs_depth')


if __name__ == '__main__':
    use()
    t = ea.tables()
    a = ea.umaps(t, TAG)
    b = ea.depth_panel(t, WIDTH - sum(ea.UMAP_W.values()) - C_W - 0.02, TAG)
    c = variance(t, C_W)
    values = ab.values_by_cluster(t['rankings'], t['gsea'])
    scores = ab.levels(COMP)['genes']
    reach = ab.whisker_range([values['genes', m] for m, _, _ in scores])
    f = [genes(t, values, k, m, ylabel, ref, ylim=reach) for k, (m, ylabel, ref) in enumerate(scores)]
    shared = ab.value_range([values['pathways', m] for m in (COMP, 'ln')])
    g = [pathways(t, values, k, m, shared) for k, m in enumerate((COMP, 'ln'))]
    assemble([[('a', a[0], 0), ('', a[1], 0), ('b', b, 0), ('c', c, 0)],
              [('d', calls(WIDTH - 2 * C_W - 0.02), 0), ('e', calls_vs_cells(C_W), 0), ('f', calls_vs_depth(C_W), 0)],
              [('g', f[0], 0), ('', f[1], 0), ('h', g[0], 0), ('', g[1], 0)]], TAG, width=WIDTH)
