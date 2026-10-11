"""PBMC figure, standalone, for choosing what enters Fig 2: LN's fold change and its SE on PBMC3k and PBMC10k, assembled
from clustering.lfc_se's drawing code, with Fig 2's subcluster panels from clustering.subcluster_recursive and
clustering.subcluster_gsea.

a, CD14 Mono against the rest in PBMC3k: for the genes with the largest and the smallest LN LFC at each level of
detection, LN's LFC with its 95% interval and scanpy's LFC; b, LN's LFC in PBMC3k against PBMC10k for CD14 Mono, the
third of genes with the smallest SE and the third with the largest; c, each effect's correlation between the datasets on
the genes all three methods call up in both; d, six markers' LFC split into the detection part and the level part, in
both datasets; e, LN's top 10 markers in PBMC3k by its score and by its detection score; f, each PBMC10k subcluster's
mean statistic against its depth term; g, the share of Hallmark calls surviving depth matching. Each plot is drawn by
its own drawing code from the saved results; a change to a plot belongs in that code.

    python -m figures.pbmc_lfc      # from reproducibility/, seconds; needs clustering's lfc_se, subclustering results
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import plot_style
from clustering import lfc_se as ls
from clustering import subcluster_gsea as sg
from clustering import subcluster_recursive as sr
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from method_colors import method_color, method_label, method_order
from paths import require_input

ROW = (2.15, 2.0, 1.75)     # each row's height
TOP = 0.25                  # room above a plot's axes for its title
TAG = 'pbmc_lfc'


def read(path, what, source):
    return pd.read_csv(require_input(path, what=what, source=source))


def panel(draw, title, stem, width, row, left=PLOT['left'], bottom=PLOT['bottom'], legend=None):
    """One plot in a slot of the given width and row height, titled; its legend placed where legend says, kept as
    drawn ('keep'), or dropped."""
    h = ROW[row]
    fig = plt.figure(figsize=(width, h))
    ax = axes_at(fig, left, bottom, width - left - PLOT['right'], h - bottom - TOP)
    draw(ax)
    ax.set_title(title)
    if legend == 'keep':
        pass
    elif legend:
        ax.legend(**legend, fontsize=plot_style.SMALL, frameon=False, handlelength=1, borderaxespad=0.2)
    elif ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


def key(width, row):
    """The method colours of f and g."""
    h = ROW[row]
    fig = plt.figure(figsize=(width, h))
    methods = method_order(['ln', 't_test', 'wilcoxon'])
    fig.legend([Patch(facecolor=method_color(m)) for m in methods], [method_label(m) for m in methods],
               title='Method', loc='upper left', bbox_to_anchor=(0.05, 1 - TOP / h), alignment='left',
               fontsize=plot_style.SMALL)
    return save(fig, f'{TAG}_key')


def carry_legend(ax):
    ax.legend([Patch(facecolor='0.35'), Patch(facecolor='0.35', alpha=0.45)], ['Pearson', 'Spearman'],
              loc='upper center', bbox_to_anchor=(0.35, -0.3), ncol=2, fontsize=plot_style.SMALL, frameon=False,
              handlelength=1, borderaxespad=0.2)


if __name__ == '__main__':
    use()
    src = 'python -m clustering.lfc_se'
    t = {k: read(ls.OUT / f'{k}.csv', f'the lfc_se {k}', src)
         for k in ('forest', 'across_genes', 'carryover', 'part_examples', 'lists')}
    children = read(sr.OUT / 'children.csv', 'the recursive subclusters', 'python -m clustering.subcluster_recursive')
    survival = read(sg.OUT / 'summary.csv', 'the subcluster GSEA summary', 'python -m clustering.subcluster_gsea')

    wa, wb = 2.75, 2.05
    a = panel(lambda ax: ls.draw_forest(ax, t['forest']), f'{ls.EXAMPLE[1]} vs rest, {ls.EXAMPLE[0]}',
              f'{TAG}_a_forest', wa, 0, left=1.15, bottom=0.62,
              legend=dict(loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=2, columnspacing=0.8))
    b = panel(lambda ax: ls.draw_se_scatter(ax, t['across_genes']), f'{ls.EXAMPLE[1]}, two datasets',
              f'{TAG}_b_se_scatter', wb, 0, left=0.5, legend=dict(loc='upper left', markerscale=3))
    c = panel(lambda ax: (ls.draw_carryover(ax, t['carryover']), carry_legend(ax)), 'Same genes, 3k and 10k',
              f'{TAG}_c_carryover', WIDTH - wa - wb - 0.02, 0, left=0.95, bottom=0.62, legend='keep')
    wd = 3.05
    d = panel(lambda ax: ls.draw_parts(ax, t['part_examples']), 'LFC = detection part + level part',
              f'{TAG}_d_parts', wd, 1, left=0.5, bottom=0.62, legend=dict(loc='upper left', ncol=2))
    e = panel(lambda ax: ls.draw_lists(ax, t['lists']), f'LN top {ls.LIST_N} markers, {ls.EXAMPLE[0]}',
              f'{TAG}_e_lists', WIDTH - wd - 0.02, 1, left=0.1, bottom=0.1)
    wf = (WIDTH - 1.3) / 2
    f = panel(lambda ax: sr.draw_depth_term(ax, children), 'PBMC10k subclusters', f'{TAG}_f_depth_term', wf, 2)
    g = panel(lambda ax: sg.draw_survival(ax, survival), 'Hallmark calls', f'{TAG}_g_gsea_survival', wf, 2)
    k = key(WIDTH - 2 * wf - 0.02, 2)
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)],
              [('d', d, 0), ('e', e, 0)],
              [('f', f, 0), ('g', g, 0), ('', k, 0)]], TAG, width=WIDTH)
