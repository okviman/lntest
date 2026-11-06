"""Fig 2a at print size: the lymph node's images, 1.42 in high, the width from the images.

The Visium HD section's hires H&E with every Cluster 1 (T-cell zone) and Cluster 3 (B-cell follicle) cell filled, with
their colour key; a 40 um window where both meet, on the full-resolution H&E, the T-cell zone's 2 um spots kept (filled)
or dropped (open) at p = 0.5, a cell never left with fewer than 3 (lymphnode.depth_imbalance --thin bins), the
follicles' all kept.

Reads lymphnode/data, the full-resolution H&E included (fetch_data, lymphnode-he-fullres), and writes
figures/diagrams/fig2_lymphnode_diagram.pdf, which figures.fig2 places on the page. The PDF is committed, so assembling
Fig 2 does not need this script's 2 GB image.

    python -m figures.diagrams.fig2_lymphnode      # from reproducibility/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import tifffile
import zarr
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.image import imread
from matplotlib.patches import Patch, Rectangle

from paths import require_input

SMALL, TINY = 6, 5.5
C = {'X': '#1f77b4', 'Y': '#ff7f0e'}   # as Fig 1: blue, orange
COMPARTMENT = {'Cluster-1': C['X'], 'Cluster-3': C['Y']}
HERE = Path(__file__).resolve().parent
LN = HERE.parents[1] / 'lymphnode' / 'data'
HE = LN / 'Visium_HD_Human_Lymph_Node_FFPE_tissue_image.tif'
WINDOW_UM, P_SUB, SEED, MIN_SPOTS = 40, 0.5, 0, 3
plt.rcParams.update({'font.size': SMALL, 'pdf.fonttype': 3})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def inset(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_facecolor('none')
    a.set_xticks([])
    a.set_yticks([])
    return a


def read_crop(tif, x0, y0, x1, y1):
    """A region of the full-resolution H&E, decoding only the tiles it covers."""
    store = tifffile.imread(tif, aszarr=True)
    try:
        z = zarr.open(store, mode='r')
        full = z['0'] if isinstance(z, zarr.Group) else z
        return np.asarray(full[y0:y1, x0:x1])
    finally:
        store.close()


def lymph_node_data():
    cells = pd.read_parquet(LN / 'cluster_1_and_3_vishd_lymph.parquet', columns=['ID', 'geometry', 'Cluster'])
    geoms = shapely.from_wkb(cells['geometry'].to_numpy())
    rings = [np.asarray(shapely.get_coordinates(g)) for g in geoms]
    centres = shapely.get_coordinates(shapely.centroid(geoms))
    sf = json.load(open(LN / 'segmented_outputs/spatial/scalefactors_json.json'))
    return cells['ID'].to_numpy(), cells['Cluster'].to_numpy(), rings, centres, sf


def pick_window(clusters, centres, size):
    """The size x size px tile where the two compartments meet most evenly: the most cells of the rarer one."""
    tiles = np.floor(centres / size).astype(int)
    df = pd.DataFrame({'tx': tiles[:, 0], 'ty': tiles[:, 1], 'c': clusters})
    counts = df.groupby(['tx', 'ty', 'c']).size().unstack(fill_value=0)
    best = counts.min(axis=1).idxmax()
    return best[0] * size, best[1] * size


def lymph_node():
    ids, clusters, rings, centres, sf = lymph_node_data()
    mpp, s = sf['microns_per_pixel'], sf['tissue_hires_scalef']
    size = WINDOW_UM / mpp
    wx, wy = pick_window(clusters, centres, size)
    # the two images and their colour key only, filling the row's height
    H, base, img_h = 1.42, 0.3, 1.1
    hires = imread(LN / 'segmented_outputs/spatial/tissue_hires_image.png')
    x0, y0 = centres.min(0) - 200
    x1, y1 = centres.max(0) + 200
    c0, r0, c1, r1 = (int(v * s) for v in (x0, y0, x1, y1))
    pad = 0.1 * size
    bx0, by0, bx1, by1 = (int(v) for v in (wx - pad, wy - pad, wx + size + pad, wy + size + pad))
    ov_h = img_h
    ov_w = ov_h * (c1 - c0) / (r1 - r0)
    zx, zh = 0.02 + ov_w + 0.12, img_h
    zw = zh * (bx1 - bx0) / (by1 - by0)
    W = zx + zw + 0.02
    fig, ax = canvas(W, H)

    # the section, every Cluster 1 and Cluster 3 cell filled
    a = inset(fig, 0.02, base, ov_w, ov_h)
    a.imshow(hires[r0:r1, c0:c1], extent=(c0, c1, r1, r0), interpolation='antialiased')
    a.add_collection(PolyCollection([r * s for r in rings], facecolors=[to_rgba(COMPARTMENT[c], 0.55) for c in clusters],
                                    edgecolors='none', rasterized=True))
    frame = 12 / 72 / ov_w * (c1 - c0)  # a frame 12 pt across, so it shows at print size
    fx, fy = (wx + size / 2) * s, (wy + size / 2) * s
    a.add_patch(Rectangle((fx - frame / 2, fy - frame / 2), frame, frame, fill=False, ec='black', lw=0.6))
    a.set_xlim(c0, c1)
    a.set_ylim(r1, r0)
    a.axis('off')
    # the key as Fig 1e's: shaded squares at the overview's fill
    fig.legend([Patch(facecolor=to_rgba(COMPARTMENT[c], 0.55), edgecolor='none') for c in ('Cluster-1', 'Cluster-3')],
               ['T cells', 'Germinal center B cells'], loc='lower left', bbox_to_anchor=(0.02 / W, 0.0), frameon=False, fontsize=6,
               handlelength=1.2, borderaxespad=0.15, labelspacing=0.3)

    # the window on the full-resolution H&E: each cell's 2 um spots, kept or dropped
    crop = read_crop(HE, bx0, by0, bx1, by1)
    z = inset(fig, zx, base, zw, zh)
    z.imshow(crop, extent=(bx0, bx1, by1, by0), interpolation='antialiased')
    inside = (centres[:, 0] > bx0) & (centres[:, 0] < bx1) & (centres[:, 1] > by0) & (centres[:, 1] < by1)
    assign = pd.read_parquet(LN / 'vishd-cluster1-cluster3-2um-assignments.parquet')
    assign = assign[assign['shape_id'].isin(set(ids[inside]))]
    pos = pd.read_parquet(LN / 'binned_outputs/square_002um/spatial/tissue_positions.parquet',
                          columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'])
    spots = assign.merge(pos, on='barcode').merge(pd.DataFrame({'shape_id': ids, 'Cluster': clusters}), on='shape_id')
    half = 1.0 / mpp
    # the T-cell zone's spots kept with probability P_SUB, a cell given dropped spots back up to MIN_SPOTS; the
    # follicles' all kept
    rng = np.random.default_rng(SEED)
    kept = (spots['Cluster'] == 'Cluster-3').to_numpy() | (rng.random(len(spots)) < P_SUB)
    for _, idx in spots.groupby('shape_id').indices.items():
        short = MIN_SPOTS - kept[idx].sum()
        if short > 0:
            kept[rng.permutation(idx[~kept[idx]])[:short]] = True
    squares = [np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
               for x, y in zip(spots['pxl_col_in_fullres'], spots['pxl_row_in_fullres'])]
    face = [to_rgba(COMPARTMENT[c], 0.7) if k else (1, 1, 1, 0) for c, k in zip(spots['Cluster'], kept)]
    edge = [(0, 0, 0, 0) if k else to_rgba(COMPARTMENT[c], 0.9) for c, k in zip(spots['Cluster'], kept)]
    z.add_collection(PolyCollection(squares, facecolors=face, edgecolors=edge, linewidths=0.4, rasterized=True))
    z.set_xlim(bx0, bx1)
    z.set_ylim(by1, by0)
    z.axis('off')
    bar = 10 / mpp
    z.plot([bx0 + 0.06 * (bx1 - bx0), bx0 + 0.06 * (bx1 - bx0) + bar], [by1 - 0.07 * (by1 - by0)] * 2, color='black',
           lw=1.5, solid_capstyle='butt')
    z.text(bx0 + 0.06 * (bx1 - bx0), by1 - 0.1 * (by1 - by0), '10 µm', color='black', fontsize=TINY, va='bottom')

    return fig


if __name__ == '__main__':
    require_input(HE, what='the lymph node full-resolution H&E', source='python -m fetch_data --arm lymphnode')
    fig = lymph_node()
    fig.savefig(HERE / 'fig2_lymphnode_diagram.pdf', dpi=600)
    plt.close(fig)
    print(f"wrote {HERE / 'fig2_lymphnode_diagram.pdf'}")
