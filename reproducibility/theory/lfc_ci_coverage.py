import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_frozen import digamma, trigamma, get_intervals
from paths import fig_name, results_dir
import plot_style
from method_colors import method_color, method_label

SEED = 0


def generate_count_data(r, p, n):
    x = np.random.negative_binomial(r, p, n)
    N_plus = np.sum(x > 0)
    N_0 = n - N_plus
    return x, N_plus, N_0


def estimate_ziln_mean(x, a, b):
    _, log_mean, _ = get_intervals(np.log(x[x>0]), a, b)
    return log_mean


def generate_scaled_binomial_draws(theta, n, rng):
    N_plus = rng.binomial(n, theta)
    return N_plus / n


def generate_beta_draws(a, b):
    return np.random.beta(a, b)


def generate_ln_draws(a, b):
    mu = digamma(a) - digamma(a + b)
    sigma = np.sqrt(trigamma(a) - trigamma(a + b))
    return np.exp(np.random.normal(mu, sigma))


def visualize_fit_to_scaled_binomial():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    theta = 0.1
    n = 1000
    experiments = 10000
    scaled_binomials = np.zeros(experiments)
    betas = np.zeros_like(scaled_binomials)
    log_normals = np.zeros_like(scaled_binomials)
    normals = np.zeros_like(scaled_binomials)

    for i in range(experiments):
        scaled_binomials[i] = generate_scaled_binomial_draws(theta, n, rng)
        betas[i] = generate_beta_draws(theta * n, n * (1 - theta))
        log_normals[i] = generate_ln_draws(theta * n, n * (1 - theta))
        normals[i] = np.random.normal(theta, np.sqrt(theta * (1 - theta) / n))

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].hist(scaled_binomials, bins=100)
    ax[1].hist(betas, bins=100, color='r')
    ax[2].hist(log_normals, bins=100)
    ax[3].hist(normals, bins=100)
    plt.tight_layout()
    _save(fig, 'fit_to_scaled_binomial.png')


def visualize_lfc_normality():
    np.random.seed(SEED)
    r_treatment = 10
    r_ctrl = 20
    p = 0.01
    n = 1000
    experiments = 10000
    lfcs_ziln = np.zeros(experiments)
    for i in range(experiments):
        x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, n)
        x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, n)
        lt = estimate_ziln_mean(x_treatment, a_treatment, b_treatment)
        lc = estimate_ziln_mean(x_ctrl, a_ctrl, b_ctrl)
        lfcs_ziln[i] = lt - lc
    lfc = np.log(r_treatment * (1 - p) / p) - np.log(r_ctrl * (1 - p) / p)
    fig = plt.figure()
    plt.hist(lfcs_ziln, bins=100)
    plt.vlines(lfc, 0, 300, color='r')
    _save(fig, 'lfc_normality.png')


def lfc_coverage():
    np.random.seed(SEED)
    r_treatment = 2
    r_ctrl = 5
    p = 0.6
    x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, 1000)
    x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, 1000)
    f, ax = plt.subplots(3, 1, figsize=plot_style.figsize(1, 3))
    ax[0].set_title(f'Treatment NB ($r={r_treatment}, p={p}$)')
    ax[0].hist(x_treatment, bins=100)
    ax[1].set_title(f'Ctrl NB ($r={r_ctrl}, p={p}$)')
    ax[1].hist(x_ctrl, bins=100, color='r')

    coverage_ziln = []
    coverage_normal = []
    coverage_log1p = []
    n_list = [10, 20, 30, 50, 100, 1000]
    for n in n_list:
        experiments = 2000
        lfc_ziln_intervals = np.zeros((2, experiments))
        lfc_normal_intervals = np.zeros((2, experiments))
        lfc_log1p_intervals = np.zeros((2, experiments))
        for i in range(experiments):
            x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, n)
            x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, n)

            for model in ['lognormal', 'naive', "log1p"]:

                if model == 'log1p':
                    log_x_t, log_x_c = np.log(1 + x_treatment), np.log(1 + x_ctrl)
                    lt, lc = np.mean(log_x_t), np.mean(log_x_c)
                    se_t, se_c = np.sqrt(np.var(log_x_t) / n), np.sqrt(np.var(log_x_c) / n)
                else:
                    _, lt, se_t = get_intervals(np.log(x_treatment[x_treatment > 0]), a_treatment, b_treatment, model=model)
                    _, lc, se_c = get_intervals(np.log(x_ctrl[x_ctrl > 0]), a_ctrl, b_ctrl, model=model)

                estimated_lfc = lt - lc
                se_lfc = np.sqrt(se_t ** 2 + se_c ** 2)
                if model == 'lognormal':
                    lfc_ziln_intervals[:, i] = estimated_lfc + 1.96 * np.array([-se_lfc, se_lfc])
                elif model == 'log1p':
                    lfc_log1p_intervals[:, i] = estimated_lfc + 1.96 * np.array([-se_lfc, se_lfc])
                else:
                    lfc_normal_intervals[:, i] = np.exp(estimated_lfc) + 1.96 * np.array([-se_lfc, se_lfc])

        lfc = np.log(r_treatment * (1 - p) / p) - np.log(r_ctrl * (1 - p) / p)

        print("n:", n)
        for model in ['lognormal', 'naive', 'log1p']:
            if model == 'lognormal':
                coverage_percentage = (np.sum(
                    (lfc > lfc_ziln_intervals[0, :]) * (lfc < lfc_ziln_intervals[1, :]))) / experiments
                coverage_ziln.append(coverage_percentage)
            elif model == 'log1p':
                coverage_percentage = (np.sum(
                    (lfc > lfc_log1p_intervals[0, :]) * (lfc < lfc_log1p_intervals[1, :]))) / experiments
                coverage_log1p.append(coverage_percentage)
            else:
                coverage_percentage = (np.sum(
                    (np.exp(lfc) > lfc_normal_intervals[0, :]) * (np.exp(lfc) < lfc_normal_intervals[1, :]))) / experiments
                coverage_normal.append(coverage_percentage)
            print(model + ":", coverage_percentage)
    ax[2].plot(n_list, coverage_log1p, color=method_color('log1p'), label=method_label('log1p'))
    ax[2].plot(n_list, coverage_normal, color='0.5', label='normal')
    ax[2].hlines(0.95, n_list[0], n_list[-1], color='black', linestyle='--', label='95% coverage')
    ax[2].plot(n_list, coverage_ziln, color=method_color('ZILN'), label=method_label('ZILN'))  # on top
    ax[2].set_xlabel('$n$')
    plt.tight_layout()
    plot_style.legend_outside(f, *ax[2].get_legend_handles_labels(), y=0.33)
    _save(f, 'lfc_ci_coverage.png')

    rows = [
        {'n': n, 'model': model, 'coverage': cov}
        for model, covs in (('lognormal', coverage_ziln),
                            ('naive', coverage_normal),
                            ('log1p', coverage_log1p))
        for n, cov in zip(n_list, covs)
    ]
    out = results_dir(__file__) / 'lfc_ci_coverage.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'wrote {out}')


FIGURES = {
    'coverage': lfc_coverage,
    'binomial-fit': visualize_fit_to_scaled_binomial,
    'lfc-normality': visualize_lfc_normality,
}


def _save(fig, name):
    out = results_dir(__file__) / fig_name(name)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figure', choices=[*FIGURES, 'all'], default='coverage')
    args = ap.parse_args()
    plot_style.use()
    for name in (FIGURES if args.figure == 'all' else [args.figure]):
        FIGURES[name]()
