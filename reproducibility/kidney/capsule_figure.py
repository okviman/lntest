"""The glomerular capsules on the kidney H&E, and one capsule's 2 um spots split into two groups.

Rebuilds RECOMB Fig 2b, which was a hand composite of a matplotlib overlay and three QuPath renders
of unseeded splits, as two figures: capsule_overview.pdf and capsule_split_illustration.pdf. The split
is spot_split's: per capsule, n_A ~ Binomial(n_spots, p) spots chosen at random go to group A, the rest
to group B. A is drawn in blue and B in orange, matplotlib's first two colours.

    python -m kidney.capsule_figure      # from reproducibility/

Needs kidney/data/ from `python -m kidney.preprocess` and the full-resolution H&E,
`python -m fetch_data --arm kidney` (kidney-he-fullres, 4.3 GB).
"""

import argparse
import json
from dataclasses import dataclass

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity
import tifffile
import zarr
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.image import imread

import plot_style
from paths import data_dir, fig_name, require_input, results_dir

DATA = data_dir(__file__)
SPLIT_P = (0.5, 0.3, 0.1)
OUTLINE = '#ffd400'   # capsule outlines on the overview
BOUNDS = '#f2f2f2'    # a capsule's outer bounds in the close-ups
GROUPS = {'A': '#1f77b4', 'B': '#ff7f0e'}  # tab10 blue and orange
FILL = 0.45           # the groups' opacity over the H&E
BAND = 2.9            # overview width / height, as in the published strip


def load():
    obs = ad.read_h5ad(require_input(DATA / 'podocytes_2um.h5ad', what='the per-2um-spot capsule matrix',
                                     source='python -m kidney.preprocess'), backed='r').obs
    sf = json.load(open(require_input(DATA / 'binned_outputs/square_002um/spatial/scalefactors_json.json',
                                      what='the 2um bin scale factors', source='python -m fetch_data --arm kidney')))
    return obs[['shape_id', 'pxl_col_in_fullres', 'pxl_row_in_fullres', 'array_row', 'array_col']].to_numpy(), sf


def capsule_outlines(spots, half):
    """Each capsule's outline: the union of the 16 um bins its 2 um spots fall in, in full-resolution pixels.

    A 16 um bin is 8 x 8 spots. Each bin's box spans its spots' squares, grown by half a pixel so that
    neighbouring bins overlap and merge instead of leaving hairline gaps.
    """
    outlines = {}
    for sid in np.unique(spots[:, 0]):
        mine = spots[spots[:, 0] == sid]
        bins = (mine[:, 3] // 8) * 100000 + mine[:, 4] // 8
        boxes = []
        for b in np.unique(bins):
            xy = mine[bins == b, 1:3]
            boxes.append(shapely.box(xy[:, 0].min() - half - 0.5, xy[:, 1].min() - half - 0.5,
                                     xy[:, 0].max() + half + 0.5, xy[:, 1].max() + half + 0.5))
        outlines[sid] = shapely.union_all(boxes)
    return outlines


def draw_outline(ax, geom, scale, **kw):
    for poly in getattr(geom, 'geoms', [geom]):
        x, y = poly.exterior.xy
        ax.plot(np.asarray(x) * scale, np.asarray(y) * scale, **kw)


def scale_bar(ax, microns, px_per_micron, label):
    """A black bar in the lower left, in place of pixel-coordinate ticks."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()  # images run top to bottom, so y0 is the bottom
    w, h = x1 - x0, y0 - y1
    left, bottom = x0 + 0.04 * w, y0 - 0.05 * h
    ax.plot([left, left + microns * px_per_micron], [bottom, bottom], color='black', lw=2, solid_capstyle='butt')
    ax.text(left, bottom - 0.025 * h, label, color='black', fontsize=plot_style.SMALL, va='bottom')


def read_crop(tif, x0, y0, x1, y1):
    """A region of the full-resolution image, decoding only the tiles it covers."""
    store = tifffile.imread(tif, aszarr=True)
    try:
        z = zarr.open(store, mode='r')
        full = z['0'] if isinstance(z, zarr.Group) else z  # level 0 of a pyramid is full resolution
        return np.asarray(full[y0:y1, x0:x1])
    finally:
        store.close()


def densest_band(centres, width, height):
    """Top of the horizontal band of this height holding the most capsule centres."""
    ys = np.sort(centres[:, 1])
    counts = [np.sum((ys >= y) & (ys < y + height)) for y in ys]
    return ys[int(np.argmax(counts))]


@dataclass
class Section:
    sf: dict           # the 2 um bins' scale factors
    half: float        # half a 2 um spot, in full-resolution pixels
    outlines: dict     # shape_id -> capsule outline, in full-resolution pixels
    capsule: float     # the capsule of the close-ups
    spots: np.ndarray  # its spots' centres, in full-resolution pixels


def section(capsule=None):
    """The capsules' outlines, and the close-up capsule (default the one with the median number of spots)."""
    spots, sf = load()
    half = sf['bin_size_um'] / sf['microns_per_pixel'] / 2
    outlines = capsule_outlines(spots, half)
    if capsule is None:
        sizes = {sid: np.sum(spots[:, 0] == sid) for sid in outlines}
        capsule = min(sizes, key=lambda s: abs(sizes[s] - np.median(list(sizes.values()))))
    return Section(sf, half, outlines, capsule, spots[spots[:, 0] == capsule, 1:3])


def draw_overview(ax, sec):
    """The hires H&E, cropped to the band of the section with the most capsules, every capsule outlined."""
    sf, outlines = sec.sf, sec.outlines
    hires = imread(require_input(DATA / 'spatial/tissue_hires_image.png', what='the hires H&E',
                                 source='python -m fetch_data --arm kidney'))
    s = sf['tissue_hires_scalef']
    centres = np.array([[g.centroid.x, g.centroid.y] for g in outlines.values()])
    x0, x1 = centres[:, 0].min() - 600, centres[:, 0].max() + 600
    height = (x1 - x0) / BAND
    y0 = densest_band(centres, x1 - x0, height) - 0.1 * height
    r0, r1, c0, c1 = (int(v) for v in (y0 * s, (y0 + height) * s, x0 * s, x1 * s))
    ax.imshow(hires[max(r0, 0):r1, max(c0, 0):c1], extent=(max(c0, 0), c1, r1, max(r0, 0)),
              interpolation='antialiased')
    for geom in outlines.values():
        draw_outline(ax, geom, s, color=OUTLINE, lw=0.6)
    # the capsule of the close-ups, framed so it can be found at print size
    bx0, by0, bx1, by1 = (v * s for v in outlines[sec.capsule].bounds)
    frame = 0.6 * max(bx1 - bx0, by1 - by0)
    ax.add_patch(Rectangle((bx0 - frame, by0 - frame), bx1 - bx0 + 2 * frame, by1 - by0 + 2 * frame,
                           fill=False, edgecolor='black', lw=0.8))
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_axis_off()
    scale_bar(ax, 500, s / sf['microns_per_pixel'], '500 µm')


def capsule_crop(sec):
    """The close-up capsule on the full-resolution H&E, with a margin, and the crop's box in pixels."""
    bx0, by0, bx1, by1 = sec.outlines[sec.capsule].bounds
    pad = 0.12 * max(bx1 - bx0, by1 - by0)
    box = tuple(int(v) for v in (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad))
    tif = require_input(DATA / 'Visium_HD_Human_Kidney_FFPE_tissue_image.tif', what='the full-resolution H&E',
                        source='python -m fetch_data --arm kidney')
    return read_crop(tif, *box), box


def draw_splits(axes, sec, crop, box, seed):
    """One random split of the capsule's spots per axes, at SPLIT_P."""
    cx0, cy0, cx1, cy1 = box
    rng = np.random.default_rng(seed)
    half, mine = sec.half, sec.spots
    for ax, p in zip(axes, SPLIT_P):
        in_a = np.zeros(len(mine), dtype=bool)
        in_a[rng.choice(len(mine), rng.binomial(len(mine), p), replace=False)] = True
        ax.imshow(crop, extent=(0, cx1 - cx0, cy1 - cy0, 0), interpolation='antialiased')
        for members, color in zip((in_a, ~in_a), GROUPS.values()):
            squares = [np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
                       for x, y in mine[members] - [cx0, cy0]]
            # filled, no edges: neighbouring squares share borders, so the group drawn last would own them all
            ax.add_collection(PolyCollection(squares, facecolors=to_rgba(color, FILL), edgecolors='none',
                                             rasterized=True))  # vector squares leave seams in PDF viewers
        draw_outline(ax, shapely.affinity.translate(sec.outlines[sec.capsule], -cx0, -cy0), 1, color=BOUNDS, lw=0.8)
        ax.set_title(rf'$p_\mathrm{{split}}$ = {p:g}')
        ax.set_axis_off()
    scale_bar(axes[0], 50, 1 / sec.sf['microns_per_pixel'], '50 µm')


def split_legend(fig, **kw):
    return fig.legend([Patch(facecolor=to_rgba(c, FILL), edgecolor='none') for c in GROUPS.values()],
                      [f'Group {k}' for k in GROUPS], **kw)


def main(capsule, seed):
    sec = section(capsule)

    fig, ax = plt.subplots(figsize=(7.2, 7.2 / BAND), layout='constrained')
    draw_overview(ax, sec)
    save(fig, 'capsule_overview')

    crop, box = capsule_crop(sec)
    fig, axes = plt.subplots(1, len(SPLIT_P), figsize=(7.2, 2.6), layout='constrained')
    draw_splits(axes, sec, crop, box, seed)
    split_legend(fig, loc='outside right upper')
    save(fig, 'capsule_split_illustration', f' (capsule {sec.capsule:g}, {len(sec.spots)} spots)')


def save(fig, stem, note=''):
    out = results_dir(__file__) / fig_name(stem)
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}{note}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--capsule', type=float, default=None, help='shape_id of the capsule to split; default the median-sized one')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    plot_style.use()
    main(args.capsule, args.seed)
