"""Lymph node block: the B-cell follicles against the T-cell zone, the T-cell zone's reads thinned and the follicles
whole, so the T-cell zone's variance grows; assembled from lymphnode.depth_imbalance's and lymphnode.depth_gsea's
drawing code.

a, b, each method's AUPRC of the T-cell zone's and of the B-cell follicles' pseudo-bulk markers, 20 replicates per
condition; c, the Spearman correlation between each method's Hallmark NES and the pseudo-bulk reference's, 10
replicates; each method's mean and SD at each fraction p of the T-cell zone's reads kept (1, the unthinned
data). Each plot is drawn by its own drawing code from the saved results; a change to a plot belongs in that
code.

    python -m figures.lymphnode_story      # from reproducibility/, seconds; needs depth_imbalance's and depth_gsea's results
    python -m figures.lymphnode_story --thin bins   # the spot version, lymphnode_story_bins.pdf
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from lymphnode import depth_gsea as dg
from lymphnode import depth_imbalance as di
from paths import require_input

ROW = 2.0
TOP = 0.25
TAG = 'lymphnode_story'


def read(path, what, source):
    return pd.read_csv(require_input(path, what=what, source=source))


def panel(draw, title, stem, width, left=PLOT['left'], legend=None):
    """One plot in a slot of the given width, titled; its legend placed where legend says, or dropped."""
    fig = plt.figure(figsize=(width, ROW))
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    draw(ax)
    ax.set_title(title)
    if legend:
        ax.legend(**legend, fontsize=plot_style.SMALL, frameon=False, handlelength=1, borderaxespad=0.2)
    elif ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The lymph node block')
    parser.add_argument('--thin', choices=list(di.OUTS), default='reads', help='The read or the spot version')
    thin = parser.parse_args().thin
    tag = TAG if thin == 'reads' else f'{TAG}_{thin}'
    use()
    source = f"python -m lymphnode.{{}} --config lymphnode/config_50rep.yaml --thin {thin}"
    results = read(di.OUTS[thin] / 'results.csv', 'the depth-imbalance results', source.format('depth_imbalance'))
    gsea = read(dg.OUTS[thin] / 'agreement.csv', 'the depth-GSEA agreement', source.format('depth_gsea'))

    w = WIDTH / 3
    a = panel(lambda ax: di.draw_errorbar(ax, results[results['cluster'] == di.T_ZONE], 'auprc',
                                     f'AUPRC, markers of\nthe {di.NAMES[di.T_ZONE]}', di.XLABEL[thin]),
              'Markers, T-cell zone', f'{tag}_a_auprc_t_zone', w)
    b = panel(lambda ax: di.draw_errorbar(ax, results[results['cluster'] == di.B_FOLLICLES], 'auprc',
                                     f'AUPRC, markers of\nthe {di.NAMES[di.B_FOLLICLES]}', di.XLABEL[thin]),
              'Markers, B-cell follicles', f'{tag}_b_auprc_b_follicles', w,
              legend=dict(handles=di.method_lines(), loc='lower right'))
    c = panel(lambda ax: dg.draw_agreement(ax, gsea, thin=thin), 'Hallmark GSEA, B over T', f'{tag}_c_gsea', w)
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)]], tag, width=WIDTH)
