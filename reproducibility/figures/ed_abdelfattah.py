"""Extended Data, Abdelfattah et al. 2022 rerun: their Fig. 4d pathways under the authors' procedure and under LN,
assembled from published.abdelfattah2022's drawing code; and the same figure with the Wilcoxon's z, and with the log1p
t-test, in place of the authors' AUC ranking (ed_abdelfattah_wilcoxon_z.pdf, ed_abdelfattah_log1p.pdf).

a, the myeloid cells' UMAP coloured by the paper's clusters and by counts per cell; b, each cluster's depth relative to
the other myeloid cells; c, each cluster's expressed genes against its log2 depth ratio, the Wilcoxon AUC and LN's score;
d, its Hallmark sets' NES against it, under both; e, the NES heatmaps of their Fig. 4d pathways, one colour bar (the
published figure's range; the Wilcoxon's larger NES saturate); f, the pathways behind the interpretation the rerun
contradicts, MC3 (s-Mac1) against MC4 (MDSC), under the Wilcoxon AUC ranking (the authors' procedure) and under LN. Drawn by the arm's own code
from the saved results, at the size it prints at; a change to a panel belongs in that code.

    python -m figures.ed_abdelfattah      # from reproducibility/, seconds; needs published/results/; all three
"""

import matplotlib.pyplot as plt
import pandas as pd

import plot_style
from figures.layout import PLOT, assemble, axes_at, pdf_size, save, use
from figures.layout import ED_WIDTH as WIDTH
from paths import require_input
from published import abdelfattah2022 as ab

SLOT = WIDTH / 3
ROW, TOP = 1.75, 0.25           # a row of plots and the room above their axes for a title
ROW1 = 2.0                      # the first row's height
QUARTER = WIDTH / 4
CELL = 0.1                      # a heatmap cell's height, in inches
COL = 0.12                      # its width
HEAT = dict(labels=1.25, top=0.3, bottom=0.72, pad=0.08, cbar=0.08, cbar_labels=0.5)
UMAP = dict(left=0.3, bottom=0.25, side=1.2, cbar=0.08, cbar_gap=0.06)


def tables():
    """The arm's saved tables, as published.abdelfattah2022 reads them."""
    return ab.tables()


UMAP_W = {'cluster': UMAP['left'] + UMAP['side'] + 0.1,
          'depth': UMAP['left'] + UMAP['side'] + UMAP['cbar_gap'] + UMAP['cbar'] + 0.3}


def umaps(t, tag):
    """The UMAP by cluster and by counts per cell, each in a slot just wide enough, the second with its colour bar."""
    out = []
    for k, by in enumerate(('cluster', 'depth')):
        fig = plt.figure(figsize=(UMAP_W[by], ROW1))
        ax = axes_at(fig, UMAP['left'], UMAP['bottom'], UMAP['side'], UMAP['side'])
        cax = (axes_at(fig, UMAP['left'] + UMAP['side'] + UMAP['cbar_gap'], UMAP['bottom'], UMAP['cbar'], UMAP['side'])
               if by == 'depth' else None)
        ab.draw_umap(ax, t['umap'], by, cax=cax, cbar_label=False)
        ax.set_title('Clusters' if by == 'cluster' else 'log10 total counts')
        out.append(save(fig, f'{tag}_a{k + 1}_umap_{by}'))
    return out


def heatmaps(t, comp, tag, letter='e', shared_xlabel=False, col=COL, cbar_labels=HEAT['cbar_labels'], vmax=ab.SHARED_VMAX):
    """The comparator's and LN's heatmaps, the first with the pathway names; their shared colour bar. Each heatmap in its
    own slot, or with shared_xlabel both in one slot under one 'Cluster' label; col, a column's width, and cbar_labels,
    the room for the colour bar's labels, narrow them; vmax, the colour range."""
    order, ticks = ab.order_of(t['clusters'])
    pair = (comp, 'ln')
    grids = ab.grids(t['gsea'], order, pair)
    w, h = col * len(order), CELL * len(ab.ROWS)
    height = HEAT['bottom'] + h + HEAT['top']
    out, im = [], None
    if shared_xlabel:
        lefts = (HEAT['labels'], HEAT['labels'] + w + 2 * HEAT['pad'])
        width = lefts[1] + w + HEAT['pad']
        fig = plt.figure(figsize=(width, height))
        axes = []
        for k, (key, left) in enumerate(zip(pair, lefts)):
            axes.append(axes_at(fig, left, HEAT['bottom'], w, h))
            im = ab.draw_heatmap(axes[-1], grids[key], ticks, row_labels=k == 0, vmax=vmax)
            axes[-1].set_title(ab.TITLE[key])
        fig.canvas.draw()   # the label at the height matplotlib gives one heatmap's, centred under both
        y = axes[0].xaxis.label.get_window_extent().y0 / fig.bbox.height
        for ax in axes:
            ax.set_xlabel('')
        fig.text((lefts[0] + lefts[1] + w) / 2 / width, y, 'Cluster', ha='center', va='bottom')
        out.append(save(fig, f'{tag}_{letter}1_heatmaps'))
    else:
        for k, key in enumerate(pair):
            left = HEAT['labels'] if k == 0 else HEAT['pad']
            fig = plt.figure(figsize=(left + w + HEAT['pad'], height))
            ax = axes_at(fig, left, HEAT['bottom'], w, h)
            im = ab.draw_heatmap(ax, grids[key], ticks, row_labels=k == 0, vmax=vmax)
            ax.set_title(ab.TITLE[key])
            out.append(save(fig, f'{tag}_{letter}{k + 1}_heatmap_{key}'))
    fig = plt.figure(figsize=(HEAT['pad'] + HEAT['cbar'] + cbar_labels, height))
    ab.draw_colorbar(axes_at(fig, HEAT['pad'], HEAT['bottom'], HEAT['cbar'], h), im)
    out.append(save(fig, f'{tag}_{letter}{len(out) + 1}_colorbar'))
    return out, height


def panel(draw, title, stem, left=PLOT['left'], width=SLOT):
    """One plot in a slot of the given width, titled."""
    fig = plt.figure(figsize=(width, ROW))
    ax = axes_at(fig, left, PLOT['bottom'], width - left - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    draw(ax)
    ax.set_title(title)
    return save(fig, stem)


def depth_panel(t, width, tag):
    """Each cluster's log2 depth ratio, MC1 to MC9, in the rest of the first row."""
    fig = plt.figure(figsize=(width, ROW1))
    left, bottom = 0.6, 0.42
    ax = axes_at(fig, left, bottom, width - left - PLOT['right'], ROW1 - bottom - TOP)
    ab.draw_depth_ratio(ax, t['clusters'])
    return save(fig, f'{tag}_b_depth_ratio')


def contrast_panel(t, width, height, comp, tag, stem=None):
    """MC3 against MC4 in the contrasted pathways, one facet per method, in the rest of the heatmaps' row, its axes
    level with theirs."""
    fig = plt.figure(figsize=(width, height))
    left, gap, bottom = 0.42, 0.12, HEAT['bottom']
    w = (width - left - gap - PLOT['right']) / 2
    pair = (comp, 'ln')
    for k, m in enumerate(pair):
        ax = axes_at(fig, left + k * (w + gap), bottom, w, height - bottom - HEAT['top'])
        ab.draw_contrast(ax, t['gsea'], m, ylabel=k == 0, lim=ab.contrast_limit(t['gsea'], pair))
        ax.set_title(ab.TITLE[m])
    ax.legend(loc='upper right', fontsize=plot_style.SMALL, frameon=False, handlelength=0.8, handletextpad=0.3,
              borderaxespad=0.1)
    return save(fig, stem or f'{tag}_f_contrast')


def call_changes_panel(gsea, height, stem, comp='wilcoxon', width=2.12, col=0.085):
    """Per cluster, the Fig. 4d pathways whose GSEA call under comp stays, vanishes, reverses or appears under LN, its
    axes level with the heatmaps', the key to the right."""
    fig = plt.figure(figsize=(width, height))
    left = 0.44
    w = col * len(ab.CLUSTERS)
    ax = axes_at(fig, left, HEAT['bottom'], w, height - HEAT['bottom'] - HEAT['top'])
    ab.draw_call_changes(ax, gsea, comp=comp)
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.replace(' under', '\nunder').replace('Neither significant', 'Neither\nsignificant') for l in labels]
    fig.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=((left + w + 0.06) / width,
               1 - HEAT['top'] / height), frameon=False, fontsize=plot_style.SMALL, handlelength=1.0, borderpad=0,
               borderaxespad=0, labelspacing=0.6, handletextpad=0.4)   # the top segment first, as the bars stack
    return save(fig, stem)


def contrast_by_cluster(t, width, height, stem, comp='wilcoxon'):
    """MC3 over MC4 on one x axis of the contrasted pathways, comp's and LN's NES side by side, on one y range, level
    with the heatmaps' axes; the key below the pathway names, the axes being narrower than it."""
    fig = plt.figure(figsize=(width, height))
    left, gap = 0.42, 0.32
    h = (height - HEAT['bottom'] - HEAT['top'] - gap) / 2
    pair = (comp, 'ln')
    lim = ab.contrast_limit(t['gsea'], pair)
    for k, c in enumerate(ab.CONTRAST):
        ax = axes_at(fig, left, HEAT['bottom'] + (1 - k) * (h + gap), width - left - PLOT['right'], h)
        ab.draw_contrast_cluster(ax, t['gsea'], c, pair, lim=lim)
        ax.set_title(ab.LABEL[c])
        if k == 0:
            ax.tick_params(labelbottom=False)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(0.03, 0), fontsize=plot_style.SMALL,
               frameon=False, handlelength=0.8, handletextpad=0.3, borderaxespad=0, labelspacing=0.3)
    return save(fig, stem)


FIGURES = {'wilcoxon': 'ed_abdelfattah', 'wilcoxon_z': 'ed_abdelfattah_wilcoxon_z', 't_test': 'ed_abdelfattah_log1p'}


def build(t, comp):
    """The figure with comp set against LN."""
    tag = FIGURES[comp]
    a = umaps(t, tag)
    b = depth_panel(t, WIDTH - sum(UMAP_W.values()) - 0.02, tag)
    values = ab.values_by_cluster(t['rankings'], t['gsea'])
    cd = []
    for letter, level in (('c', 'genes'), ('d', 'pathways')):
        entries = ab.levels(comp)[level]
        shared = ab.value_range([values[level, m] for m, _, _ in entries]) if level == 'pathways' else None
        for k, (m, ylabel, ref) in enumerate(entries):
            title = ab.NAME[(comp, 'ln')[k]]
            cd.append((letter if k == 0 else '', panel(
                lambda ax, level=level, m=m, ylabel=ylabel, ref=ref, k=k, shared=shared: ab.draw_by_depth(
                    ax, values[level, m], t['clusters'], m, ylabel, ref, ylim=shared, show_ylabel=shared is None or k == 0,
                    points=level == 'pathways'),
                title, f'{tag}_{letter}{k + 1}_{level}_{m}', left=0.5, width=QUARTER)))
    e, height = heatmaps(t, comp, tag)
    used = sum(pdf_size(x)[0] for x in e)
    f = contrast_panel(t, WIDTH - used - 0.02, height, comp, tag)
    assemble([[('a', a[0], 0), ('', a[1], 0), ('b', b, 0)],
              [(l, x, 0) for l, x in cd],
              [('e', e[0], 0), ('', e[1], 0), ('', e[2], 0), ('f', f, 0)]], tag, width=WIDTH)


if __name__ == '__main__':
    use()
    t = tables()
    for comp in ab.COMPARATORS:
        build(t, comp)
