import argparse

import numpy as np

from paths import results_dir
import pandas as pd
import statsmodels.stats.multitest as smm
from lntest import get_LN_lfcs
from baselines import get_test_results, scanpy_sig_test


nx = 10000
ny = 10000
n_genes = 1500
columns = ["method", "accuracy", "precision", "recall", "tpr", "tnr", "fpr", "fnr", "f1", "dispersion", "rep_no"]
methods = ["LN", "t-test", 'wilcoxon']

# RECOMB Table 2's design is 'sparse', Table 1's 'dense'.
SETTINGS = {
    'sparse': {'non_de_mu': 10, 'd2': 1.0, 'd1s': [1., 1.5, 2.]},
    'dense': {'non_de_mu': 100, 'd2': 0.1, 'd1s': [0.1, 0.2, 1.]},
}


def de_shift(setting):
    """Upward shift of a DEG's mean in group 1: |N(0, 5)| when sparse, N(15, 5) when dense."""
    if setting == 'sparse':
        return np.abs(np.random.normal(0, 5, (1, n_genes)))
    return np.random.normal(15, 5, (1, n_genes))


def out_dir(setting):
    return results_dir(__file__) / f"nde_mu{SETTINGS[setting]['non_de_mu']}"


def run(setting, seed=0):
    np.random.seed(seed)
    non_de_mu, d2 = SETTINGS[setting]['non_de_mu'], SETTINGS[setting]['d2']

    # Create an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)

    for d1 in SETTINGS[setting]['d1s']:
        # each gene is DE in group 1 with probability 0.1, and then goes up or down with equal
        # probability by the same fold change, so group 1's library size, which CP10K divides
        # by, barely moves; up-only DEGs shift every other gene after normalisation
        rep_count = 20
        for rep in range(rep_count):
            z1 = np.random.binomial(1, 0.1, (1, n_genes))
            fold = 1 + de_shift(setting) / non_de_mu
            sign = 2 * np.random.binomial(1, 0.5, (1, n_genes)) - 1
            mu1 = non_de_mu * fold ** (sign * z1)
            mu2 = non_de_mu

            r1 = 1 / d1
            r2 = 1 / d2
            p1 = 1 / (1 + d1 * mu1)
            p2 = 1 / (1 + d2 * mu2)

            X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
            Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

            for method in methods:
                if method == "LN":
                    _, LN_p_vals = get_LN_lfcs(Y, X, test='t')
                    adj_pvals = smm.multipletests(LN_p_vals, alpha=0.05, method='bonferroni')[1]
                else:
                    _, adj_pvals = scanpy_sig_test(X, Y, method=method)
                    method = "Scanpy " + method

                results = get_test_results(adj_pvals, z1, verbose=False)
                results["method"] = method
                results["rep_no"] = rep
                results["dispersion"] = d1
                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

    out = out_dir(setting)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"d1_vs_d2_01_nde_mu_{int(non_de_mu)}.csv", index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Planted-DEG NB simulations with the designs of RECOMB Tables 1 and 2."
    )
    ap.add_argument('--setting', default='sparse', choices=list(SETTINGS),
                    help="'sparse' (non-DE mean 10) is Table 2's design; 'dense' (non-DE mean 100) Table 1's")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run(args.setting, args.seed)
