"""Extended Data Fig 6: the Visium HD lymph node in full, assembled from the arm's drawing code.

Germinal center B cells (the B-cell follicles) against T cells (the T-cell zone), the T cells thinned. a-c: Fig 2's
spot version (the T cells' 2 um spots kept with probability p_sub, the B cells' all; lymphnode.depth_imbalance --thin
bins), each method's AUPRC of the T cells' (a) and of the B cells' (b) pseudo-bulk markers and the calls outside the
reference markers (c), 20 replicates per p_sub. d-f: the read version (each T-cell read kept with probability p_sample;
--thin reads), the AUPRC of each cell type's markers (20 replicates) and the Spearman correlation of each method's
Hallmark NES with the pseudo-bulk reference's (lymphnode.depth_gsea, 10 replicates). Titles name the cell type only; the
x axis tells spots (p_sub) from reads (p_sample). As Fig 1's error bars, the mean over the replicates with bars of one
SD. One method key, in b. Each plot is drawn by its own drawing code from the saved results, at the size it prints at;
a change to a plot belongs in that code.

    python -m figures.ed6      # from reproducibility/, a few seconds; needs depth_imbalance's and depth_gsea's results
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from figures.layout import PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from lymphnode import depth_gsea as dg
from lymphnode import depth_imbalance as di
from paths import require_input

SLOT = WIDTH / 3
ROW, TOP = 2.0, 0.25    # a row of plots and the room above their axes for a title


def read(path, what, source):
    return pd.read_csv(require_input(path, what=what, source=source))


def panel(draw, title, stem, legend=None, width=SLOT):
    """One plot in a slot of the given width, a third of the page by default, titled; its legend placed where legend
    says, or dropped."""
    fig = plt.figure(figsize=(width, ROW))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], width - PLOT['left'] - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    draw(ax)
    ax.set_title(title)
    if legend:
        ax.legend(**legend, fontsize=plot_style.SMALL, frameon=False, handlelength=1, borderaxespad=0.2)
    elif ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


def markers(results, cluster, thin):
    return lambda ax: di.draw_errorbar(ax, results[results['cluster'] == cluster], 'auprc', 'AUPRC', di.XLABEL[thin])


if __name__ == '__main__':
    use()
    source = 'python -m lymphnode.{} --config lymphnode/config_50rep.yaml --thin {}'
    results = {thin: read(di.OUTS[thin] / 'results.csv', f'the depth-imbalance results ({thin})',
                          source.format('depth_imbalance', thin)) for thin in ('bins', 'reads')}
    gsea = read(dg.OUTS['reads'] / 'agreement.csv', 'the depth-GSEA agreement (reads)', source.format('depth_gsea', 'reads'))

    spots = results['bins']
    a, b = (panel(markers(spots, cl, 'bins'), title, f'ed6_{letter}_spots_auprc_{stem}',
                  legend=dict(handles=di.method_lines(), loc='lower right') if letter == 'b' else None)
            for letter, cl, title, stem in (('a', di.T_ZONE, 'T cells', 't_zone'), ('b', di.B_FOLLICLES, 'B cells', 'b_follicles')))
    c = panel(lambda ax: di.draw_errorbar(ax, di.per_task_false_calls(spots), 'false_calls',
                                          'Calls outside the\nreference markers', di.XLABEL['bins']),
              'Calls outside the reference', 'ed6_c_spots_false_calls')
    d = panel(markers(results['reads'], di.T_ZONE, 'reads'), 'T cells', 'ed6_d_reads_auprc_t_zone')
    e = panel(markers(results['reads'], di.B_FOLLICLES, 'reads'), 'B cells', 'ed6_e_reads_auprc_b_follicles')
    f = panel(lambda ax: dg.draw_agreement(ax, gsea, thin='reads', ylabel='NES agreement'), 'Hallmark GSEA',
              'ed6_f_reads_gsea')
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)], [('d', d, 0), ('e', e, 0), ('f', f, 0)]], 'ed6', width=WIDTH)
