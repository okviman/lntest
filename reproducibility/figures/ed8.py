"""Extended Data Fig 8: PBMC10k in full, Fig 2's PBMC dataset, assembled from the clustering arm's existing figures.

a and b: avg_jaccard for all significant genes and for the top 20, with one key (clustering.plots.draw_box); c and d: the
Jaccard heatmaps for all significant genes and for the top 20, one row per method and one column per resolution, each grid on the colour scale its
methods share over every resolution, as the arm's jaccard_heatmaps figures (draw_heatmap). The two grids share
columns, so the cell types are named under d only. Each is drawn by the arm's own drawing code from the saved
metrics, at the size it prints at; a change to it belongs in that code.

    python -m figures.ed8      # from reproducibility/, a few seconds; needs clustering's metrics
"""

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from clustering import plots as cp
from figures.layout import KEY_W, MAX_H, PLOT, ROW_GAP, TEXT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from method_colors import method_label, method_order
from paths import require_input, results_dir

CLUSTERING = results_dir(cp.__file__)
PBMC = CLUSTERING / 'pbmc10k'
# the grids' margins, in inches: a heading and the resolutions above, the cell types below c only
GRID = dict(left=0.4, heading=0.15, top=0.18, bottom=0.84, below=0.05, cbar=0.8, hgap=0.08, vgap=0.06)
LABEL = 5   # the cell types, in points, as Fig 2i
A_H, A_TOP = 1.9, 0.25   # a's row and the room above its axes, for its title, as Fig 2's PBMC row


def jaccard_box(df, order, cluster_count, title, stem, key):
    """One Jaccard box plot in half the page less the key; with the key beside it if key."""
    slot = (WIDTH - KEY_W) / 2
    fig = plt.figure(figsize=(slot + (KEY_W if key else 0), A_H))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], slot - PLOT['left'] - PLOT['right'], A_H - PLOT['bottom'] - A_TOP)
    cp.draw_box(ax, df, 'avg_jaccard_sig', cp.YLABELS['avg_jaccard_sig'], title, order, cluster_count)
    if not key:
        ax.get_legend().remove()
    return save(fig, stem)


def heatmap_grid(groups, tag, keys, cluster_count, heading, h, stem, cell_types):
    """Every method's heatmaps of this tag at every resolution, h inches tall each, with the grid's colour bar."""
    methods = method_order([method_label(m) for m in cp.KNOWN_METHODS if (tag, m) in groups])
    by_label = {method_label(m): m for m in cp.KNOWN_METHODS}
    vmax = cp.shared_vmax(groups, tag)
    n_rows, n_cols = len(methods), len(keys)
    bottom = GRID['bottom'] if cell_types else GRID['below']
    height = GRID['heading'] + GRID['top'] + n_rows * h + (n_rows - 1) * GRID['vgap'] + bottom
    w = (WIDTH - GRID['left'] - GRID['cbar'] - (n_cols - 1) * GRID['hgap']) / n_cols
    fig = plt.figure(figsize=(WIDTH, height))
    axes = []
    for r, label in enumerate(methods):
        _, mats = groups[(tag, by_label[label])]
        y = bottom + (n_rows - 1 - r) * (h + GRID['vgap'])
        axes.append([axes_at(fig, GRID['left'] + c * (w + GRID['hgap']), y, w, h) for c in range(n_cols)])
        for ax, rk in zip(axes[-1], keys):
            cp.draw_heatmap(ax, mats[rk], vmax, '', cluster_numbers=False, label_size=LABEL)
    cp.label_heatmap_grid(axes, [m.replace(' ', '\n', 1) for m in methods],  # two lines, to fit beside a row
                          [f'{cluster_count(rk)} clusters' for rk in keys], cell_types=cell_types)
    fig.text((GRID['left'] + (n_cols * w + (n_cols - 1) * GRID['hgap']) / 2) / WIDTH, 1 - 0.02 / height, heading,
             ha='center', va='top', fontsize=TEXT)
    grid_h = n_rows * h + (n_rows - 1) * GRID['vgap']
    cax = axes_at(fig, WIDTH - GRID['cbar'] + 0.06, bottom, 0.08, grid_h)
    fig.colorbar(ScalarMappable(cmap='viridis', norm=Normalize(0, vmax)), cax=cax, label='Jaccard Index')
    return save(fig, stem)


if __name__ == '__main__':
    use()
    config = yaml.safe_load(open(require_input(CLUSTERING.parent / 'config_pbmc10k.yaml', what='the PBMC10k clustering config',
                                               source='the clustering arm')))
    metrics = PBMC / 'metrics'
    keys, order, _, cluster_count = cp.resolution_axis(str(metrics), config)
    top = config['top_genes']
    boxes = {tag: pd.read_csv(require_input(metrics / f'avg_jaccard_{tag}_by_resolution.csv', what=f'the PBMC10k {tag} Jaccard',
                                            source='python -m clustering.metrics')) for tag in ('allsig', f'top{top}')}
    groups = cp.heatmap_groups(str(metrics), 'jaccard_matrix', keys)
    n_rows = len({m for _, m in groups})
    # the heatmaps' height: what the page leaves once a, the margins and the gaps are placed
    fixed = A_H + 2 * ROW_GAP + 2 * (GRID['heading'] + GRID['top'] + (n_rows - 1) * GRID['vgap']) \
        + GRID['below'] + GRID['bottom']
    h = (MAX_H - fixed) / (2 * n_rows)
    a = jaccard_box(boxes['allsig'], order, cluster_count, cp.jaccard_title('allsig', top), 'ed8_a_pbmc10k_jaccard_allsig',
                    key=False)
    b = jaccard_box(boxes[f'top{top}'], order, cluster_count, cp.jaccard_title('top', top), f'ed8_b_pbmc10k_jaccard_top{top}',
                    key=True)
    c = heatmap_grid(groups, 'allsig', keys, cluster_count, cp.jaccard_title('allsig', top), h,
                     'ed8_c_pbmc10k_jaccard_heatmaps_allsig', cell_types=False)
    d = heatmap_grid(groups, f'top{top}', keys, cluster_count, cp.jaccard_title('top', top), h,
                     f'ed8_d_pbmc10k_jaccard_heatmaps_top{top}', cell_types=True)
    assemble([[('a', a, 0), ('b', b, 0)], [('c', c, 0)], [('d', d, 0)]], 'ed8', width=WIDTH)
