import argparse

import numpy as np
import pandas as pd

from paths import data_dir

# Was unseeded: every run produced different counts AND a different lfcs.csv,
# so the ground truth itself moved. Seeded here.
SEED = 0

N_LIST = [10, 20, 30, 50, 100, 1000, 2000]
N_GENES = 1000


def generate_nb_counts(r1, r2, p1, p2, n_samples):
    counts_1 = np.random.negative_binomial(r1, p1, n_samples)
    counts_2 = np.random.negative_binomial(r2, p2, n_samples)
    return counts_1, counts_2


def get_nb_lfc_data(root, seed=SEED):
    """Fixed-LFC variant: every tenth gene is differentiated by r1 vs r2."""
    np.random.seed(seed)
    r1 = 2
    r2 = 5
    p = 0.6

    ds_path = root / f"data_from_NB_parameters_r1_{r1}_r2_{r2}_p_0{int(10 * p)}"

    for n_samples in N_LIST:
        counts_1 = np.zeros((n_samples, N_GENES))
        counts_2 = np.zeros_like(counts_1)
        for g in range(N_GENES):
            if g % 10 == 0:
                # differentiated gene
                r_variable = r1
            else:
                # non-differentiated gene
                r_variable = r2
            counts_1[:, g], counts_2[:, g] = generate_nb_counts(r_variable, r2, p, p, n_samples)

        out = ds_path / f"n_{n_samples}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(counts_1).to_csv(out / "treatment_counts.csv")
        pd.DataFrame(counts_2).to_csv(out / "control_counts.csv")

    print(f"wrote {ds_path}")


def get_random_lfc_data(root, seed=SEED):
    """Random-LFC variant: every tenth gene gets an LFC drawn from U(-4, 4)."""
    np.random.seed(seed)
    r2 = 5
    p = 0.6

    ds_path = root / f"data_from_NB_parameters_r2_{r2}_p_0{int(10 * p)}".replace('.', '')

    for n_samples in N_LIST:
        counts_1 = np.zeros((n_samples, N_GENES))
        counts_2 = np.zeros_like(counts_1)
        lfcs = np.zeros(N_GENES)
        for g in range(N_GENES):
            if g % 10 == 0:
                # differentiated gene
                lfc = np.random.uniform(-4, 4)
                lfcs[g] = lfc
                # lfc = log(r1 * (1 - p) / p) - log(r2 * (1 - p) / p) = log(r1 / r2) --> r1 = exp(lfc + log(r2))
                r_variable = 2 ** (lfc + np.log2(r2))
            else:
                # non-differentiated gene
                r_variable = r2
            counts_1[:, g], counts_2[:, g] = generate_nb_counts(r_variable, r2, p, p, n_samples)

        out = ds_path / f"n_{n_samples}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(counts_1).to_csv(out / "treatment_counts.csv")
        pd.DataFrame(counts_2).to_csv(out / "control_counts.csv")
        pd.DataFrame(lfcs).to_csv(out / "lfcs.csv")

    print(f"wrote {ds_path}")


DATASETS = {"random-lfc": get_random_lfc_data, "fixed-lfc": get_nb_lfc_data}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=[*DATASETS, 'all'], default='random-lfc')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    # data/, not the working directory: this writes hundreds of MB and data/ is
    # gitignored. Was `path = ''`, i.e. wherever you happened to be standing.
    root = data_dir(__file__)
    root.mkdir(parents=True, exist_ok=True)
    for name in (DATASETS if args.dataset == 'all' else [args.dataset]):
        DATASETS[name](root, args.seed)
