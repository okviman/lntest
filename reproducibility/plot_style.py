"""Text sizes and axes shared by every arm's figures.

Sizes are in points at the size a figure is drawn, so each figure is drawn at
roughly its printed size (a panel about PANEL inches) and included without scaling;
then the text is the same size in every figure of the paper.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from method_colors import legend_sorted

TEXT = 8    # axis labels, tick labels, titles
SMALL = 6   # legends and annotations
PANEL = (2.6, 2.2)  # width, height of one axes, in inches
# An axes with more dots than this has them rasterized; save such figures at RASTER_DPI
MANY_POINTS = 5000
RASTER_DPI = 300
# A mean with error bars, the same in every figure that draws one
ERRORBAR = dict(marker='o', ms=3, lw=1.3, capsize=2, capthick=0.8, elinewidth=0.8)

RC = {
    'font.size': TEXT,
    'axes.titlesize': TEXT,
    'axes.labelsize': TEXT,
    'xtick.labelsize': TEXT,
    'ytick.labelsize': TEXT,
    'legend.fontsize': SMALL,
    'legend.title_fontsize': SMALL,
    'figure.titlesize': TEXT,
    'figure.labelsize': TEXT,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'xtick.bottom': True,
    'ytick.left': True,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#b0b0b0',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
}


def use():
    """Apply the shared text sizes, spines and legend style to every figure drawn after this call."""
    plt.rcParams.update(RC)


def figsize(ncols=1, nrows=1, extra_width=0.0):
    """Figure size for an nrows x ncols grid of panels, plus room for an outside legend."""
    return (PANEL[0] * ncols + extra_width, PANEL[1] * nrows)


def strip(ax, x, values, color, width, rng):
    """Every observation as a jittered point over its box, as the clustering box plots draw them."""
    values = list(values)
    ax.scatter(x + rng.uniform(-width / 4, width / 4, len(values)), values, s=4, color=color,
               edgecolors='0.2', linewidths=0.4, alpha=0.8, zorder=3)


def legend_outside(fig, handles=None, labels=None, title=None, y=None):
    """One legend for the whole figure, just right of its axes; by default the first axes' entries.

    Methods are listed in METHODS order whatever order they were drawn in. `y` is in figure
    coordinates and defaults to the top of the axes, below any titles. A figure's only legend
    needs no title; give one only to tell two keys apart.
    """
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    handles, labels = legend_sorted(handles, labels)
    for ax in fig.axes:
        ax.apply_aspect()  # a fixed aspect moves the axes only once applied
    boxes = [ax.get_position() for ax in fig.axes if ax.get_visible() and ax.axison]
    x = max(b.x1 for b in boxes) + 0.01
    y = max(b.y1 for b in boxes) if y is None else y
    return fig.legend(handles, labels, title=title, loc='upper left', bbox_to_anchor=(x, y),
                      alignment='left')


def rasterize_dense(fig):
    """Rasterize the dots, and their error bars, of every axes holding more than MANY_POINTS dots.

    Axes, text and lines stay vector, so the PDF stays sharp without carrying thousands of paths.
    """
    from matplotlib.collections import LineCollection, PathCollection
    for ax in fig.axes:
        dots = [c for c in ax.collections if isinstance(c, PathCollection)]
        markers = [l for l in ax.lines if l.get_linestyle() in ('None', '', ' ')
                   and l.get_marker() not in (None, 'None', '', ' ')]
        n = sum(len(c.get_offsets()) for c in dots) + sum(len(l.get_xdata()) for l in markers)
        if n <= MANY_POINTS:
            continue
        bars = [c for c in ax.collections if isinstance(c, LineCollection) and len(c.get_segments()) > MANY_POINTS]
        for artist in dots + markers + bars:
            artist.set_rasterized(True)


def boxplot_grid(ax):
    """Box plots keep only the horizontal grid lines: their x-axis is categorical."""
    ax.grid(False, axis='x')


def zero_row(mean, sd):
    """Where zero goes on a log axis: one decade below the smallest positive mean or lower bar."""
    mean, sd = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
    positive = np.concatenate([mean[mean > 0], (mean - sd)[mean - sd > 0]])
    return 10.0 ** (np.floor(np.log10(positive.min())) - 1)


def errorbar_log(ax, x, mean, sd, zero, **kwargs):
    """Mean with one-SD bars on a log axis; a zero mean, or a bar reaching zero, ends on the zero row."""
    m, s = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
    y = np.where(m > 0, m, zero)
    lo = np.where(m - s > 0, m - s, zero)
    hi = np.where(m + s > 0, m + s, zero)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], **ERRORBAR, **kwargs)


def log_axis_with_zero(ax, zero, largest):
    """A log y-axis whose bottom tick, labelled 0, is the zero row, set off by a break mark on the spine."""
    ax.set_yscale('log')
    top = 10.0 ** np.ceil(np.log10(largest))
    ax.set_ylim(zero / 3, top)
    decades = 10.0 ** np.arange(np.log10(zero) + 1, np.log10(top) + 1)
    ax.yaxis.set_major_locator(FixedLocator(np.concatenate([[zero], decades])))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: '0' if np.isclose(v, zero) else f'$10^{{{int(round(np.log10(v)))}}}$'))
    gap = zero * np.sqrt(10)
    for f in (0.8, 1.25):
        ax.plot([-0.025, 0.025], [gap * f / 1.25, gap * f * 1.25], transform=ax.get_yaxis_transform(),
                color='black', lw=0.8, clip_on=False)
