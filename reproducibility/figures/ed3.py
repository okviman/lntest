"""Extended Data Fig 3: kidney spot splitting in full, assembled from the arm's existing figures.

a: split_degs_tpr_fpr, b: split_degs_ap_pr_auc, c: split_degs_pr_curves_grid (kidney.spot_split_plots), each with
its method key and the grid in its own number of columns. Each is drawn by its own drawing code from the saved
results, at the size it prints at; a change to a panel belongs in that code.

    python -m figures.ed3      # from reproducibility/, a few seconds; needs kidney's spot-splitting results
"""

import json

import pandas as pd

from figures.kidney_panels import pair, pr_curve_grid
from figures.layout import assemble, use
from figures.layout import ED_WIDTH as WIDTH
from kidney import spot_split_plots as ssp
from kidney.fpr_plots import SPLIT
from paths import require_input, results_dir

KIDNEY = results_dir(ssp.__file__)


if __name__ == '__main__':
    use()
    df = pd.read_csv(require_input(KIDNEY / 'split_degs_results.csv', what='the with-DEG spot-splitting results',
                                   source='python -m kidney.spot_split (see README)'))
    curve_data = json.load(open(require_input(KIDNEY / 'split_degs_pr_curve_data.json', what='its PR curves',
                                              source='python -m kidney.spot_split (see README)')))
    a = pair(ssp.draw_tpr_fpr, df, SPLIT, 'ed3_a_split_tpr_fpr', width=WIDTH)
    b = pair(ssp.draw_ap_pr_auc, df, SPLIT, 'ed3_b_split_ap_pr_auc', width=WIDTH)
    c = pr_curve_grid(curve_data, SPLIT, 'ed3_c_split_pr_curves_grid', width=WIDTH)
    assemble([[('a', a, 0)], [('b', b, 0)], [('c', c, 0)]], 'ed3', width=WIDTH)
