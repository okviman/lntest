"""Extended Data Fig 4: kidney spot splitting in full, assembled from the arm's existing figures. a: Fig 1f's FPR on the
no-DEG null with MAST, over the first 3 splits of each p_split (kidney.split_null_mast); b: the share of null genes
with uncorrected p below 0.05 against p_split (kidney.null_calibration.draw_calibration; a calibrated test gives 0.05,
dashed); one method key; c: split_degs_tpr_fpr, d: split_degs_pr_curves_grid (kidney.spot_split_plots), each with its
method key and the grid in its own number of columns; the AUPRC is Fig 1g. Each is drawn by its own drawing code from
the saved results, at the size it prints at; a change to a panel belongs in that code.

    python -m figures.ed4      # from reproducibility/, a few seconds; needs kidney's spot-splitting results
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from figures.kidney_panels import calibration, null_fpr, pair, pr_curve_grid
from figures.layout import KEY_W, ROW_H, assemble, method_key, save, use
from figures.layout import ED_WIDTH as WIDTH
from kidney import spot_split_plots as ssp
from kidney import split_null_mast as snm
from kidney.fpr_plots import SPLIT
from paths import require_input, results_dir

KIDNEY = results_dir(ssp.__file__)
SLOT = (WIDTH - KEY_W) / 2
TOP = 0.25   # a and b's title space, as kidney_panels draws them


def key(methods, stem):
    """The method colours, level with the plots' tops."""
    fig = plt.figure(figsize=(KEY_W, ROW_H))
    method_key(fig, methods, 0.05, 1 - TOP / ROW_H)
    return save(fig, stem)


if __name__ == '__main__':
    use()
    df = pd.read_csv(require_input(KIDNEY / 'split_degs_results.csv', what='the with-DEG spot-splitting results',
                                   source='python -m kidney.spot_split (see README)'))
    curve_data = json.load(open(require_input(KIDNEY / 'split_degs_pr_curve_data.json', what='its PR curves',
                                              source='python -m kidney.spot_split (see README)')))
    saved = pd.read_csv(require_input(snm.PER_SPLIT, what='the no-DEG spot-splitting splits',
                                      source='python -m kidney.spot_split (see README)'))
    mast = pd.read_csv(require_input(snm.CSV, what="MAST's spot-splitting FPR", source='python -m kidney.split_null_mast'))
    fpr = snm.table(saved, mast)
    a = null_fpr(fpr, SPLIT, 'Spot splitting, no DEGs', 'ed4_a_split_null_fpr_mast', SLOT)
    b = calibration('spot splitting', 'Calibration, no DEGs', 'ed4_b_split_calibration', SLOT)
    c = pair(ssp.draw_tpr_fpr, df, SPLIT, 'ed4_c_split_tpr_fpr', width=WIDTH)
    d = pr_curve_grid(curve_data, SPLIT, 'ed4_d_split_pr_curves_grid', width=WIDTH)
    assemble([[('a', a, 0), ('b', b, 0), ('', key(fpr['method'].unique(), 'ed4_key'), 0)],
              [('c', c, 0)], [('d', d, 0)]], 'ed4', width=WIDTH)
