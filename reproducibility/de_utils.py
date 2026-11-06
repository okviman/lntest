"""
Shared utilities for differential expression in the reproducibility folder.

Provides: layer preparation (counts, norm_counts, log1p_norm), LN test, t-test,
wilcoxon, and MAST. Used by celltype, clustering, and subsampling scripts.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from lntest import rank_genes_groups_ln

# Optional R/MAST
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    _HAS_RPY2 = True
except ImportError:
    _HAS_RPY2 = False


def tocsr_if_not_sparse(x):
    """Convert ndarray/dense matrix to CSR sparse format if not already sparse."""
    if sp.issparse(x):
        return x.copy()
    return sp.csr_matrix(x)


def prepare_adata_layers(adata):
    """
    Ensure adata has layers: counts, norm_counts, log1p_norm.
    Assumes adata.X is raw counts. Modifies adata in place.
    """
    if "counts" not in adata.layers:
        adata.layers["counts"] = tocsr_if_not_sparse(adata.X)
    if "norm_counts" not in adata.layers:
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        adata.layers["norm_counts"] = tocsr_if_not_sparse(temp.X)
        del temp
    if "log1p_norm" not in adata.layers:
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        adata.layers["log1p_norm"] = tocsr_if_not_sparse(temp.X)
        del temp
    return adata


def run_ln_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="norm_counts",
    groups=None,
    reference="rest",
    sparse=True,
    corr_method="bonferroni",
    **kwargs,
):
    """
    Run LN test. Uses lntest.rank_genes_groups_ln.
    Default reference='rest' gives one-vs-rest; set groups and reference for two-group.
    """
    rank_genes_groups_ln(
        adata,
        groupby,
        key_added=key_added,
        layer=layer,
        groups=groups,
        reference=reference,
        sparse=sparse,
        corr_method=corr_method,
        **kwargs,
    )


def run_ttest_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="log1p_norm",
    use_raw=False,
    groups=None,
    reference="rest",
    corr_method="bonferroni",
    **kwargs,
):
    """Run Scanpy t-test on layer (default log1p_norm)."""
    _groups = "all" if groups is None else groups
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="t-test",
        corr_method=corr_method,
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
        groups=_groups,
        reference=reference,
        **kwargs,
    )


def run_wilcoxon_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="log1p_norm",
    use_raw=False,
    groups=None,
    reference="rest",
    corr_method="bonferroni",
    tie_correct=True,
    **kwargs,
):
    """Run Scanpy wilcoxon on layer (default log1p_norm), tie-corrected as Seurat's."""
    _groups = "all" if groups is None else groups
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon",
        corr_method=corr_method,
        tie_correct=tie_correct,
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
        groups=_groups,
        reference=reference,
        **kwargs,
    )


def run_mast_de(
    adata,
    group_col,
    group_1=None,
    group_2=None,
    log1p_layer="log1p_norm",
    key_added="mast",
    cdr=True,
):
    """
    Run MAST differential expression via rpy2.

    - cdr: include the cellular detection rate as a covariate (~ group + cngeneson), as MAST's vignette does;
      False fits ~ group, as Seurat's FindMarkers(test.use="MAST") does by default.

    - If group_1 and group_2 are provided: two-group comparison (group_1 vs group_2).
      Writes result to adata.uns[key_added] and returns it.
    - If both are None: one-vs-rest per cluster. Returns dict[cluster_str, DataFrame]
      with columns: names, logfoldchanges, scores, pvals, pvals_adj.

    Requires rpy2 and R package MAST.
    """
    if not _HAS_RPY2:
        raise ImportError("run_mast_de requires rpy2. Install with: pip install rpy2")
    importr("MAST")
    ro.globalenv["use_cdr"] = ro.BoolVector([cdr])

    if log1p_layer not in adata.layers:
        raise ValueError(f"Layer '{log1p_layer}' not found in adata.layers")

    if group_1 is not None and group_2 is not None:
        # Two-group: subset cells and use their expression matrix
        mask = adata.obs[group_col].isin([group_1, group_2])
        ad_sub = adata[mask]
        X_sub = ad_sub.layers[log1p_layer]
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        cell_names_sub = list(ad_sub.obs_names)
        gene_names_sub = list(ad_sub.var_names)
        n_cells_sub, n_genes_sub = X_sub.shape
        X_T_sub = np.asfortranarray(X_sub.T)
        r_expr_sub = ro.r.matrix(
            ro.FloatVector(X_T_sub.ravel(order="F")),
            nrow=n_genes_sub,
            ncol=n_cells_sub,
            byrow=False,
        )
        r_expr_sub.rownames = ro.StrVector(gene_names_sub)
        r_expr_sub.colnames = ro.StrVector(cell_names_sub)
        ro.globalenv["expr_mat"] = r_expr_sub
        group_labels = ad_sub.obs[group_col].values
        binary_labels = np.where(group_labels == group_1, "target", "rest")
        cell_data = pd.DataFrame(
            {"wellKey": cell_names_sub, "group": binary_labels},
            index=cell_names_sub,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_cell_data = ro.conversion.py2rpy(cell_data)
        ro.globalenv["cell_data"] = r_cell_data

        ro.r("""
        library(MAST)
        library(data.table)
        sca <- FromMatrix(
            exprsArray = expr_mat,
            cData = data.frame(cell_data),
            fData = data.frame(primerid = rownames(expr_mat))
        )
        colData(sca)$group <- factor(colData(sca)$group, levels = c("rest", "target"))
        cdr <- colSums(assay(sca) > 0)
        colData(sca)$cngeneson <- scale(cdr)
        # every gene detected in every cell (dense synthetic data) leaves the detection rate constant and unscalable
        zlmCond <- if (use_cdr && sd(cdr) > 0) zlm(~ group + cngeneson, sca) else zlm(~ group, sca)
        summaryCond <- summary(zlmCond, doLRT = "grouptarget")
        summaryDt <- summaryCond$datatable
        fcHurdle <- merge(
            summaryDt[component == "H", .(primerid, `Pr(>Chisq)`)],
            summaryDt[component == "logFC" & contrast == "grouptarget", .(primerid, coef)],
            by = "primerid"
        )
        colnames(fcHurdle) <- c("names", "pvals", "logfoldchanges")
        fcHurdle$pvals_adj <- p.adjust(fcHurdle$pvals, method = "bonferroni")
        fcHurdle$scores <- -log10(fcHurdle$pvals + 1e-300) * sign(fcHurdle$logfoldchanges)
        mast_result <- fcHurdle
        rm(sca, zlmCond, summaryCond, summaryDt, fcHurdle)
        gc()
        """)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.globalenv["mast_result"])
        df = df.set_index("names")
        df = df.reindex(adata.var_names)
        result_dict = {
            "names": df.index.values,
            "logfoldchanges": df["logfoldchanges"].values,
            "scores": df["scores"].values,
            "pvals": df["pvals"].values,
            "pvals_adj": df["pvals_adj"].values,
        }
        de_key = f"{group_1}_vs_{group_2}"
        adata.uns[key_added] = {de_key: result_dict}
        ro.r("rm(expr_mat, cell_data, mast_result); gc()")
        return adata.uns[key_added]

    # One-vs-rest: loop over clusters (full adata)
    # R/MAST FromMatrix expects a dense matrix; we convert once and pass to R, then drop Python copies to limit memory
    X = adata.layers[log1p_layer]
    if sp.issparse(X):
        X = X.toarray()
    cell_names = list(adata.obs_names)
    gene_names = list(adata.var_names)
    n_cells, n_genes = X.shape
    X_T = np.asfortranarray(X.T)
    r_expr = ro.r.matrix(
        ro.FloatVector(X_T.ravel(order="F")),
        nrow=n_genes,
        ncol=n_cells,
        byrow=False,
    )
    r_expr.rownames = ro.StrVector(gene_names)
    r_expr.colnames = ro.StrVector(cell_names)
    ro.globalenv["expr_mat"] = r_expr
    del X, X_T  # free dense copies; R holds the data

    # One-vs-rest: loop over clusters
    cluster_labels = adata.obs[group_col].values
    unique_clusters = sorted(pd.unique(cluster_labels))
    results = {}
    for target_cluster in unique_clusters:
        binary_labels = np.where(cluster_labels == target_cluster, "target", "rest")
        cell_data = pd.DataFrame(
            {"wellKey": cell_names, "group": binary_labels},
            index=cell_names,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_cell_data = ro.conversion.py2rpy(cell_data)
        ro.globalenv["cell_data"] = r_cell_data
        ro.r("""
        library(MAST)
        library(data.table)
        sca <- FromMatrix(
            exprsArray = expr_mat,
            cData = data.frame(cell_data),
            fData = data.frame(primerid = rownames(expr_mat))
        )
        colData(sca)$group <- factor(colData(sca)$group, levels = c("rest", "target"))
        cdr <- colSums(assay(sca) > 0)
        colData(sca)$cngeneson <- scale(cdr)
        # every gene detected in every cell (dense synthetic data) leaves the detection rate constant and unscalable
        zlmCond <- if (use_cdr && sd(cdr) > 0) zlm(~ group + cngeneson, sca) else zlm(~ group, sca)
        summaryCond <- summary(zlmCond, doLRT = "grouptarget")
        summaryDt <- summaryCond$datatable
        fcHurdle <- merge(
            summaryDt[component == "H", .(primerid, `Pr(>Chisq)`)],
            summaryDt[component == "logFC" & contrast == "grouptarget", .(primerid, coef)],
            by = "primerid"
        )
        colnames(fcHurdle) <- c("names", "pvals", "logfoldchanges")
        fcHurdle$pvals_adj <- p.adjust(fcHurdle$pvals, method = "bonferroni")
        fcHurdle$scores <- -log10(fcHurdle$pvals + 1e-300) * sign(fcHurdle$logfoldchanges)
        mast_result <- fcHurdle
        rm(sca, zlmCond, summaryCond, summaryDt, fcHurdle)
        gc()
        """)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.globalenv["mast_result"])
        df = df.set_index("names")
        df = df.reindex(gene_names)
        df["names"] = df.index
        results[str(target_cluster)] = df[["names", "logfoldchanges", "scores", "pvals", "pvals_adj"]].copy()
    ro.r("rm(expr_mat, cell_data, mast_result); gc()")
    return results


def run_mast_two_groups(X, Y):
    """MAST through Seurat's FindMarkers on two count matrices, rows the observations: Y against X after NormalizeData
    (log1p CP10K), with the gene filters off (logfc.threshold = 0, min.pct = 0) so that every gene is tested. Returns
    FindMarkers' p_val, its p_val_adj (Bonferroni over every gene) and avg_log2FC, in gene order."""
    if not _HAS_RPY2:
        raise ImportError("run_mast_two_groups requires rpy2. Install with: pip install rpy2")
    importr("Seurat")
    counts = np.vstack([X, Y]).astype(float)
    genes = [f'g{j}' for j in range(counts.shape[1])]   # no underscores, which Seurat rewrites
    m = ro.r.matrix(ro.FloatVector(np.asfortranarray(counts.T).ravel(order='F')), nrow=counts.shape[1],
                    ncol=counts.shape[0])
    m.rownames = ro.StrVector(genes)
    m.colnames = ro.StrVector([f'c{i}' for i in range(counts.shape[0])])
    ro.globalenv["counts_mat"] = m
    ro.globalenv["groups"] = ro.StrVector(['X'] * len(X) + ['Y'] * len(Y))
    ro.r("""
    suppressPackageStartupMessages(library(Seurat))
    obj <- CreateSeuratObject(counts = as(counts_mat, "dgCMatrix"))
    obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)
    obj$group <- groups
    Idents(obj) <- "group"
    fm <- FindMarkers(obj, ident.1 = "Y", ident.2 = "X", test.use = "MAST", logfc.threshold = 0, min.pct = 0,
                      verbose = FALSE)
    fm$names <- rownames(fm)
    rm(obj, counts_mat); invisible(gc())
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(ro.globalenv["fm"])
    df = df.set_index("names").reindex(genes)
    ro.r("rm(fm, groups); invisible(gc())")
    return {"names": np.array(genes), "logfoldchanges": df["avg_log2FC"].values, "pvals": df["p_val"].values,
            "pvals_adj": df["p_val_adj"].values}
