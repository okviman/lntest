"""PBMC block after the marker decision (LN's markers are its significant up genes ranked by the detection score, gene
rankings are its score), assembled from clustering.multires_depth's, clustering.lfc_se's and clustering.marker_rule's
drawing code.

a, b, every Leiden cluster at five resolutions against the rest, PBMC10k and PBMC3k: its median score over its expressed
genes against its log2 depth ratio, one colour per method; c, each rule's top N markers in PBMC3k found again in
PBMC10k's top N, LN's markers and LN's score in LN's colour, the log1p t-test and the Wilcoxon in theirs (the colours of
every marker panel); d, LN's top 10 markers in PBMC3k by its score and by its marker rule. e-i, every cluster at the five
resolutions, mean per resolution of its top 20: e, f, Monaco et al. 2019's immune housekeeping genes; g, h, genes the
Human Protein Atlas (25.1) calls enriched for the cluster's majority type; i, PBMC10k, the genes of the cluster's
protein markers (13 antibodies, AUC >= 0.75) that are in the top 20; j, the same per protein at the resolutions of at
least 18 clusters, with the number of clusters where it is a marker. Each plot is drawn by its own drawing code from the
saved results; a change to a plot belongs in that code.

    python -m figures.pbmc_story      # from reproducibility/, seconds; needs multires_depth's, lfc_se's and marker_rule's results
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from clustering import lfc_se as ls
from clustering import marker_rule as mk
from clustering import multires_depth as md
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from paths import require_input

ROW = (2.05, 2.0, 2.0)
TOP = 0.25
TAG = 'pbmc_story'


def read(path, what, source, **kw):
    return pd.read_csv(require_input(path, what=what, source=source), **kw)


def panel(draw, title, stem, width, row, left=PLOT['left'], bottom=PLOT['bottom'], legend=None):
    """One plot in a slot of the given width and row height, titled; its legend placed where legend says, or dropped."""
    h = ROW[row]
    fig = plt.figure(figsize=(width, h))
    ax = axes_at(fig, left, bottom, width - left - PLOT['right'], h - bottom - TOP)
    draw(ax)
    ax.set_title(title)
    if legend:
        ax.legend(**legend, fontsize=plot_style.SMALL, frameon=False, handlelength=1, borderaxespad=0.2)
    elif ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


if __name__ == '__main__':
    use()
    comps = read(md.OUT / 'comparisons.csv', 'the multires_depth comparisons', 'python -m clustering.multires_depth',
                 dtype={'cluster': str})
    mc = read(ls.OUT / 'marker_concordance.csv', 'the marker concordance', 'python -m clustering.lfc_se')
    lists = read(ls.OUT / 'lists.csv', 'the marker lists', 'python -m clustering.lfc_se')
    per = read(mk.OUT / 'per_list.csv', 'the marker_rule per-list table', 'python -m clustering.marker_rule',
               dtype={'cluster': str})
    genes = read(mk.OUT / 'protein_genes.csv', 'the marker_rule protein genes', 'python -m clustering.marker_rule')

    w, q, r = 2.3, WIDTH / 4, 1.62
    a = panel(lambda ax: md.draw_depth(ax, comps, 'PBMC10k'), 'PBMC10k, all resolutions', f'{TAG}_a_pbmc10k_depth', w,
              0, left=0.7, legend=dict(loc='upper left', markerscale=1.5))
    b = panel(lambda ax: md.draw_depth(ax, comps, 'PBMC3k'), 'PBMC3k, all resolutions', f'{TAG}_b_pbmc3k_depth', w, 0,
              left=0.7)
    c = panel(lambda ax: ls.draw_marker_concordance(ax, mc), 'Markers, PBMC3k to PBMC10k', f'{TAG}_c_markers',
              WIDTH - 2 * w - 0.02, 0, left=0.7, legend=dict(loc='upper left', ncol=2, columnspacing=0.6))
    d = panel(lambda ax: ls.draw_lists(ax, lists), f'LN top {ls.LIST_N} markers, PBMC3k', f'{TAG}_d_lists',
              WIDTH - 2 * q, 1, left=0.1, bottom=0.1)
    by_res = []
    for letter, metric, name, row in (('e', 'housekeeping', 'PBMC10k', 1), ('f', 'housekeeping', 'PBMC3k', 1),
                                      ('g', 'hpa', 'PBMC10k', 2), ('h', 'hpa', 'PBMC3k', 2),
                                      ('i', 'protein_recall', 'PBMC10k', 2)):
        by_res.append(panel(lambda ax, m=metric, n=name: mk.draw_by_resolution(ax, per, n, m), name,
                            f'{TAG}_{letter}_{name.lower()}_{metric}', q if row == 1 else r, row, left=0.62))
    e, f, g, h, i = by_res
    j = panel(lambda ax: mk.draw_protein_genes(ax, genes), 'By protein, PBMC10k', f'{TAG}_j_protein_genes',
              WIDTH - 3 * r, 2, left=0.9)
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)], [('d', d, 0), ('e', e, 0), ('f', f, 0)],
              [('g', g, 0), ('h', h, 0), ('i', i, 0), ('j', j, 0)]], TAG, width=WIDTH)
