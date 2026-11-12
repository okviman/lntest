"""The B-cell follicles against the T-cell zone of the lymph node, with the reads of the T-cell zone thinned and the
follicles whole, a figure outside the paper. Assembled from the drawing code of lymphnode.depth_imbalance and
lymphnode.depth_gsea.

a, b, the AUPRC of each method for the pseudo-bulk markers of the T-cell zone (a) and of the B-cell follicles (b), over
20 replicates per condition.
c, the Spearman correlation between the Hallmark NES of each method and of the pseudo-bulk reference, over 10 replicates.
In each panel, the mean and SD of each method at each fraction p of the reads of the T-cell zone kept (1 for the
unthinned data).

    python -m figures.lymphnode_story               # from reproducibility/, seconds; needs the results of depth_imbalance and depth_gsea
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
