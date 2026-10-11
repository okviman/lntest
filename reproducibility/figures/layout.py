"""Shared layout of the assembled figures: 183 mm wide (Extended Data 180 mm), at most 170 mm tall, every panel placed at 100%, letters laid over them.

Panels are drawn at the size they print at, so their text prints at the size it was drawn: TEXT for axis
labels, ticks and titles, plot_style's SMALL for legends. Axes are placed in inches (axes_at), so panels in
different files line up. assemble() places each row's panel PDFs left to right, top-aligned, the rows top to
bottom, and stops if a row is wider than the figure.

Outputs go to figures/results/: paper/ the figures the paper prints (PAPER), other/ every other assembled figure,
panels/ the parts they are assembled from.
"""

import io

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

import plot_style
from method_colors import method_color, method_label, method_order
from paths import results_dir

WIDTH = 7.2      # inches: 183 mm, Nature's double column
ED_WIDTH = 180 / 25.4   # inches: Extended Data pages are at most 180 mm wide
MAX_H = 170 / 25.4      # inches: the tallest figure, main or Extended Data
TEXT = 7         # axis labels, ticks and titles; legends keep plot_style.SMALL
LETTER_PT = 8    # panel letters, bold
LETTER = 0.15    # blank strip left of a panel without a margin of its own, for its letter
ROW_GAP = 0.1
PLOT = dict(left=0.6, bottom=0.42, top=0.13, right=0.1)  # margins around a plot's axes
ROW_H = 1.7      # a row of plots with PLOT's margins
KEY_W = 0.9      # a method key right of a row's last axes
PAPER = ('fig1', 'fig2', *(f'ed{k}' for k in range(1, 11)))   # the assembled figures the paper prints


def use():
    """plot_style's figure style, with TEXT in place of its axis text size."""
    plot_style.use()
    plt.rcParams.update({k: TEXT for k, v in plot_style.RC.items() if v == plot_style.TEXT and 'size' in k})


def axes_at(fig, left, bottom, width, height, **kw):
    """An axes placed in inches from the figure's lower left."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def out_dir(sub):
    """figures/results/<sub>, made if missing."""
    d = results_dir(__file__) / sub
    d.mkdir(exist_ok=True)
    return d


def save(fig, name):
    # PDF whatever LNTEST_FIG_FORMAT says: assemble() places PDF pages
    out = out_dir('panels') / f'{name}.pdf'
    fig.savefig(out, dpi=600)  # no tight bounding box: the file is its slot
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f'wrote {out} ({w:.2f} x {h:.2f} in)')
    return out


def pdf_size(path):
    """A one-page PDF's width and height, in inches."""
    box = PdfReader(path).pages[0].mediabox
    return float(box.width) / 72, float(box.height) / 72


def method_key(fig, methods, x, y, ncol=1, loc='upper left'):
    """The method colours, as the errorbar plots draw them, placed by loc at (x, y) in figure coordinates."""
    return fig.legend([Line2D([], [], color=method_color(m), marker='o', ms=plot_style.ERRORBAR['ms'],
                              lw=plot_style.ERRORBAR['lw']) for m in method_order(methods)],
                      [method_label(m) for m in method_order(methods)], title='Method',
                      loc=loc, bbox_to_anchor=(x, y), alignment='left', ncol=ncol)


def assemble(rows, name, width=WIDTH, gap=ROW_GAP):
    """Place each row's panels (letter, PDF, strip left of it) at 100%, and letter each at its top left."""
    heights = [max(pdf_size(path)[1] for _, path, _ in row) for row in rows]
    total = sum(heights) + gap * (len(rows) - 1)
    page = PageObject.create_blank_page(width=width * 72, height=total * 72)
    letters = plt.figure(figsize=(width, total))
    top = total
    for row, height in zip(rows, heights):
        x = 0
        for letter, path, strip in row:
            src = PdfReader(path).pages[0]
            w, h = pdf_size(path)
            # pypdf clips a merged page to its own box, so nothing outside a panel's page shows
            page.merge_transformed_page(src, Transformation().translate(
                (x + strip) * 72 - float(src.mediabox.left), (top - h) * 72 - float(src.mediabox.bottom)))
            letters.text(x / width, top / total, letter, fontsize=LETTER_PT, fontweight='bold', ha='left', va='top')
            x += strip + w
        if x > width + 0.01:
            raise SystemExit(f"row {''.join(l for l, _, _ in row)} is {x:.2f} in wide, more than {width:.2f}")
        top -= height + gap
    overlay = io.BytesIO()
    letters.savefig(overlay, format='pdf', transparent=True)
    plt.close(letters)
    page.merge_page(PdfReader(overlay).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    out = out_dir('paper' if name in PAPER else 'other') / f'{name}.pdf'
    with open(out, 'wb') as fh:
        writer.write(fh)
    print(f'wrote {out} ({width:.2f} x {total:.2f} in, {width * 25.4:.0f} x {total * 25.4:.0f} mm)')
    if total > MAX_H + 0.01:
        print(f'  WARNING: {total * 25.4:.0f} mm tall, more than {MAX_H * 25.4:.0f} mm')
