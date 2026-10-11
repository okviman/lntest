"""Extended Data Fig 7: the PBMC multi-resolution clustering beyond Fig 2, assembled from clustering.multires_depth's,
clustering.multires_gaps' and clustering.marker_rule's drawing code.

Every Leiden cluster at five resolutions against the rest. a, PBMC3k: the variance ratio to the rest at equal mean, a
box at each number of clusters, every cluster a point, as Fig 2e for PBMC10k; b, both datasets: each cluster's variance
ratio against its |log2 depth ratio|, with the Spearman correlation; c, d, PBMC10k and PBMC3k: each test's median score
over the cluster's expressed genes against its log2 depth ratio, with its least-squares line and Pearson r. e, PBMC3k:
the share of each rule's top 20 that are Monaco et al. 2019's immune housekeeping genes, as Fig 2f for PBMC10k; f, g,
PBMC10k and PBMC3k: the share of the top 20 that the Human Protein Atlas (25.1) calls enriched for the cluster's
majority type; e-g the mean over each resolution's clusters with bars of one standard error. The clusters' DEGs and their
effect sizes are Extended Data Fig 8. The method key beside d, the key of the marker rules beside g. Each plot is drawn
by its own drawing code from the saved results, at the size it prints at; a change to a plot belongs in that code.

    python -m figures.ed7      # from reproducibility/, seconds; needs multires_depth's, multires_gaps' and marker_rule's results
    python -m figures.ed7 --pbmc10k-only   # the version without PBMC3k, ed7_pbmc10k.pdf: PBMC10k's median score
                                           # against depth, variance ratio against |log2 depth ratio|, HPA share,
                                           # and the key
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_style
from clustering import lfc_se as ls
from clustering import marker_rule as mk
from clustering import multires_depth as md
from clustering import multires_gaps as mg
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from method_colors import method_color, method_label
from paths import require_input

ROW = (2.05, 2.0, 1.9)
TOP = 0.25


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


KEY_W = 0.95   # a key's slot, right of d and of g


def key(entries, row, stem):
    """A key in a slot of its own, its entries from the top: (colour, alpha, label) each."""
    h = ROW[row]
    fig = plt.figure(figsize=(KEY_W, h))
    line = dict(marker='o', ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'])
    labels = [l.replace(' (score rank)', '\n(score rank)') for _, _, l in entries]   # fits the slot
    fig.legend([Line2D([], [], color=c, alpha=a, **line) for c, a, _ in entries], labels,
               loc='upper left', bbox_to_anchor=(0, 1 - TOP / h), frameon=False, fontsize=plot_style.SMALL,
               handlelength=1.4, borderaxespad=0)
    return save(fig, stem)


METHODS = [(method_color(m), 1.0, method_label(m)) for m in ('ln', 't_test', 'wilcoxon')]
RULES = [(*ls.MARKER_STYLE[r], ls.MARKER_RULES[r]) for r in mk.RULES]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended Data Fig 7')
    parser.add_argument('--pbmc10k-only', action='store_true', help='Without PBMC3k: ed7_pbmc10k.pdf')
    only = parser.parse_args().pbmc10k_only
    use()
    comps = read(md.OUT / 'comparisons.csv', 'the multires_depth comparisons', 'python -m clustering.multires_depth',
                 dtype={'cluster': str})
    gaps = read(mg.OUT / 'comparisons.csv', 'the multires_gaps comparisons', 'python -m clustering.multires_gaps')
    per = read(mk.OUT / 'per_list.csv', 'the marker_rule per-list table', 'python -m clustering.marker_rule',
               dtype={'cluster': str})

    q = (WIDTH - KEY_W) / 3
    if only:
        tag = 'ed7_pbmc10k'
        a = panel(lambda ax: md.draw_depth(ax, comps, 'PBMC10k'), 'PBMC10k', f'{tag}_a_depth', q, 0, left=0.55)
        b = panel(lambda ax: mg.draw_gap_vs_depth(ax, gaps, names=['PBMC10k']), 'PBMC10k', f'{tag}_b_gap_vs_depth', q, 0,
                  left=0.62, legend=dict(loc='upper left', handletextpad=0.2))
        c = panel(lambda ax: mk.draw_by_resolution(ax, per, 'PBMC10k', 'hpa'), 'PBMC10k', f'{tag}_c_hpa', q, 0, left=0.62)
        assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0), ('', key(RULES, 0, f'{tag}_key'), 0)]], tag, width=WIDTH)
        raise SystemExit
    a_w, b_w = 1.45, 1.7   # b wide enough for its title and legend; c and d share the rest of the row
    cd = (WIDTH - KEY_W - a_w - b_w - 0.02) / 2
    a = panel(lambda ax: mg.draw_gaps(ax, gaps, 'PBMC3k', 'gap_equal_mean'), 'PBMC3k', 'ed7_a_pbmc3k_gap', a_w, 0, left=0.5)
    b = panel(lambda ax: mg.draw_gap_vs_depth(ax, gaps), 'PBMC10k and PBMC3k', 'ed7_b_gap_vs_depth', b_w, 0, left=0.62,
              legend=dict(loc='upper left', handletextpad=0.2))
    c, d = (panel(lambda ax, n=name: md.draw_depth(ax, comps, n), name, f'ed7_{letter}_{name.lower()}_depth', cd, 0,
                  left=0.55) for letter, name in (('c', 'PBMC10k'), ('d', 'PBMC3k')))
    r = (WIDTH - KEY_W) / 3
    e, f, g = (panel(lambda ax, m=metric, n=name: mk.draw_by_resolution(ax, per, n, m), name,
                     f'ed7_{letter}_{name.lower()}_{metric}', r, 1, left=0.62)
               for letter, metric, name in (('e', 'housekeeping', 'PBMC3k'), ('f', 'hpa', 'PBMC10k'), ('g', 'hpa', 'PBMC3k')))
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0), ('d', d, 0), ('', key(METHODS, 0, 'ed7_key_methods'), 0)],
              [('e', e, 0), ('f', f, 0), ('g', g, 0), ('', key(RULES, 1, 'ed7_key_rules'), 0)]], 'ed7', width=WIDTH)
