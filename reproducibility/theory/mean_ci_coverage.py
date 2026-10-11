import numpy as np
import pandas as pd

from utils_frozen import get_intervals_synthetic_data
from paths import results_dir


if __name__ == '__main__':
    mu = 0
    sigma_2 = 1.5
    theta = 0.1
    n = 1000
    n_experiments = 2000

    true_mean = theta * np.exp(mu + 0.5 * sigma_2)

    rows = []
    for label, model in (("LN", "lognormal"), ("NAIVE", "naive")):
        intervals, estimated_means = get_intervals_synthetic_data(
            mu, sigma_2, theta, experiments=n_experiments, n=n, model=model
        )
        coverage = np.sum(
            (true_mean > intervals[0, :]) * (true_mean < intervals[1, :])
        ) / n_experiments
        rmse = np.sqrt(np.mean((true_mean - estimated_means) ** 2))

        print(f"{label} RESULTS:")
        print("Coverage percentage (ideal = 0.95): ", coverage)
        print("RMSE to the true mean: ", rmse)

        rows.append({"model": model, "coverage": coverage, "rmse": rmse,
                     "n": n, "experiments": n_experiments, "true_mean": true_mean})

    out = results_dir(__file__) / "mean_ci_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
