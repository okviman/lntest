"""Rows of kidney panels shared by the assembled figures, for either experiment: spot splitting (p_split) or UMI
downsampling (p_sample). Each panel is drawn by the arm's own drawing code; these only place it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_style
from figures.layout import KEY_W, PLOT, ROW_H, WIDTH, axes_at, method_key, save
from kidney import null_calibration as nc
from kidney import spot_split_plots as ssp
from kidney.fpr_plots import draw_fpr_log
from method_colors import method_draw_order
from paths import require_input, results_dir

GRID_TITLE, GRID_ASPECT, GRID_GAP = 0.3, 0.8, 0.22   # the PR-curve grid: title space per row, axes height over width, gap


def null_auc_curves(null_df, results, curve_data, p_symbol, stems, width=WIDTH):
    """Three panels with equal axes: the null FPR, the AUPRC (from results) and the PR curves, one method key and the p key
    beside the last."""
    curves, p_values = curve_data['curves'], np.array(curve_data['p_values'])
    methods = method_draw_order(curves[next(iter(curves))])

    slot = (width - KEY_W) / 3
    aw = slot - PLOT['left'] - PLOT['right']
    ah = ROW_H - PLOT['bottom'] - PLOT['top']

    def plot_axes(width):
        fig = plt.figure(figsize=(width, ROW_H))
        return fig, axes_at(fig, PLOT['left'], PLOT['bottom'], aw, ah)

    fig, ax = plot_axes(slot)
    draw_fpr_log(ax, null_df, p_symbol)
    fpr = save(fig, stems[0])

    fig, ax = plot_axes(slot)
    ssp.draw_auprc(ax, results, p_symbol)
    auc = save(fig, stems[1])

    fig, ax = plot_axes(slot + KEY_W)
    selected_p, styles = ssp.draw_pr_curves(ax, curves, methods, p_values)
    x = (PLOT['left'] + aw + 0.05) / (slot + KEY_W)
    y = 1 - PLOT['top'] / ROW_H
    method_key(fig, methods, x, y)
    fig.legend([Line2D([], [], color='black', lw=1.3, ls=styles[p]) for p in selected_p],
               [f'{p:.2g}' for p in selected_p], title=p_symbol,
               loc='upper left', bbox_to_anchor=(x, y - 0.36), alignment='left')
    return fpr, auc, save(fig, stems[2])


def pair(draw, df, p_symbol, stem, width=WIDTH):
    """A two-panel figure of spot_split_plots, drawn by its draw function, with its key."""
    slot = (width - KEY_W) / 2
    fig = plt.figure(figsize=(width, ROW_H))
    aw, ah = slot - PLOT['left'] - PLOT['right'], ROW_H - PLOT['bottom'] - PLOT['top']
    draw([axes_at(fig, k * slot + PLOT['left'], PLOT['bottom'], aw, ah) for k in range(2)], df, p_symbol)
    plot_style.legend_outside(fig)
    return save(fig, stem)


def pr_curve_grid(curve_data, p_symbol, stem, width=WIDTH):
    """The PR curves at every p, one panel each, in the grid's own columns, labelled on its outer edge, with its key."""
    p_values, curves = np.array(curve_data['p_values']), curve_data['curves']
    methods = method_draw_order(curves[next(iter(curves))])
    n_cols = min(ssp.GRID_COLS, len(p_values))
    n_rows = (len(p_values) + n_cols - 1) // n_cols
    aw = (width - KEY_W - PLOT['left'] - PLOT['right'] - (n_cols - 1) * GRID_GAP) / n_cols
    ah = GRID_ASPECT * aw
    height = PLOT['bottom'] + n_rows * (ah + GRID_TITLE)
    fig = plt.figure(figsize=(width, height))
    axes = []
    for i, p in enumerate(p_values):
        r, c = divmod(i, n_cols)
        ax = axes_at(fig, PLOT['left'] + c * (aw + GRID_GAP), PLOT['bottom'] + (n_rows - 1 - r) * (ah + GRID_TITLE), aw, ah)
        ssp.draw_pr_curve(ax, curves, p, methods, p_symbol)
        axes.append(ax)
    ssp.label_grid_outer(axes, n_cols)
    plot_style.legend_outside(fig)
    return save(fig, stem)


CALIBRATION_ALPHA = 0.05


def null_fpr(df, p_symbol, title, stem, width):
    """The no-DEG FPR as Fig 1f draws it (kidney.fpr_plots.draw_fpr_log), in a slot of the given width, titled."""
    fig = plt.figure(figsize=(width, ROW_H))
    top = 0.25
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], width - PLOT['left'] - PLOT['right'], ROW_H - PLOT['bottom'] - top)
    draw_fpr_log(ax, df, p_symbol)
    ax.set_title(title)
    if ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


def calibration(experiment, title, stem, width):
    """The share of null genes with uncorrected p below CALIBRATION_ALPHA on one of the kidney's no-DEG nulls
    (kidney.null_calibration), in a slot of the given width, titled; a calibrated test gives the dashed line."""
    df = pd.read_csv(require_input(results_dir(nc.__file__) / nc.CSV, what='the kidney null calibration',
                                   source='python -m kidney.null_calibration'))
    fig = plt.figure(figsize=(width, ROW_H))
    top = 0.25
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], width - PLOT['left'] - PLOT['right'], ROW_H - PLOT['bottom'] - top)
    nc.draw_calibration(ax, df, experiment, CALIBRATION_ALPHA)
    ax.set_title(title)
    if ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)
