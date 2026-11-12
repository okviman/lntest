import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import confusion_matrix



def scanpy_sig_test(X, Y, method='t-test', normalization='CP10K', corr_method="bonferroni", return_raw=False,
                    tie_correct=True):
    """LFCs and corrected p-values per gene, in gene order; the uncorrected p-values too if return_raw.
    Bonferroni and a tie-corrected Wilcoxon, Seurat's defaults; tie_correct is read by the Wilcoxon only."""
    nx = X.shape[0]
    ny = Y.shape[0]
    n_genes = Y.shape[1]
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    if normalization == 'median-of-ratios':
        X = X.astype(float)
        Y = Y.astype(float)
        X_ = X.copy()
        Y_ = Y.copy()
        X_[X_ <= 0] = np.nan
        Y_[Y_ == 0] = np.nan

        denom_Y = np.exp(np.nanmean(np.log(Y_), 0))
        c_Y = np.nanmedian(Y_ / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X_), 0))
        c_X = np.nanmedian(X_ / denom_X, 1, keepdims=True)
        X /= c_X

        Y[Y == np.nan] = 0.
        X[X == np.nan] = 0.

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    adata.layers["counts"] = adata.X.copy()
    if normalization == 'CP10K':
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy's t-test
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X", corr_method=corr_method,
                            tie_correct=tie_correct)

    # Extract DE Results
    de_results = pd.DataFrame({
        "gene": adata.uns['rank_genes_groups']['names']['Y'],
        "log2_fc": adata.uns["rank_genes_groups"]["logfoldchanges"]["Y"],
        "p_value": adata.uns["rank_genes_groups"]["pvals"]["Y"],
        "p_adj": adata.uns["rank_genes_groups"]["pvals_adj"]["Y"]
    })
    de_results["gene"] = [int(gene.replace("Gene", "")) for gene in de_results["gene"]]
    gene_idx_sorted = np.argsort(de_results["gene"])

    if return_raw:
        return (de_results["log2_fc"][gene_idx_sorted], de_results["p_adj"][gene_idx_sorted],
                de_results["p_value"][gene_idx_sorted])
    return de_results["log2_fc"][gene_idx_sorted], de_results["p_adj"][gene_idx_sorted]


def get_test_results(adj_p_vals, true_lfcs, verbose=True):
    gt_sig_idx = (true_lfcs != 0).reshape((-1,))
    pred_sig_idx = (adj_p_vals < 0.05)
    conf_mat = confusion_matrix(gt_sig_idx, pred_sig_idx)
    tn, fp, fn, tp = conf_mat.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.sum((gt_sig_idx == pred_sig_idx)) / gt_sig_idx.size

    if verbose:
        # Print results
        print(f"TPR: {tpr:.2f}")
        print(f"TNR: {tnr:.2f}")
        print(f"FPR: {fpr:.2f}")
        print(f"FNR: {fnr:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print()
    results = {
        "f1": f1, "accuracy": accuracy, "recall": recall, "precision": precision,
        "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr
    }
    return results

