import numpy as np
import scipy.sparse as sp

from ._ln_test import get_LN_lfcs, get_LN_lfcs_sparse


def _to_dense(a):
    """Convert (possibly sparse) matrix to a dense numpy array."""
    if sp is not None and sp.issparse(a):
        return a.toarray()
    # anndata can sometimes give numpy matrix; force ndarray
    return np.asarray(a)

CORR_METHODS = ("benjamini-hochberg", "bonferroni")


def _adjust_pvalues(pvals, method, n_genes):
    """Adjusted p-values, by the same route scanpy takes."""
    if method == "benjamini-hochberg":
        from statsmodels.stats.multitest import multipletests

        pvals = np.asarray(pvals, dtype=float).copy()
        # scanpy replaces NaN with 1 before correcting; an all-zero gene can
        # produce one, and multipletests would otherwise propagate it.
        pvals[np.isnan(pvals)] = 1
        return multipletests(pvals, alpha=0.05, method="fdr_bh")[1]
    if method == "bonferroni":
        return np.minimum(np.asarray(pvals, dtype=float) * n_genes, 1.0)
    raise ValueError(f"corr_method must be one of {set(CORR_METHODS)}.")


def rank_genes_groups_ln(
    adata,
    groupby: str,
    groups=None,
    reference: str = "rest",
    n_genes: int | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    key_added: str = "rank_genes_groups",
    test: str = "t",  # forwarded to get_LN_lfcs
    rankby_abs: bool = False,
    sparse: bool = True,
    corr_method: str = "bonferroni",
):
    """
    Takes normalized data and performs LN's t-test. Updates the adata object with the results.

    corr_method
        Multiple-testing correction, using scanpy's vocabulary and scanpy's
        implementation: ``"benjamini-hochberg"`` or ``"bonferroni"``.

        The default differs from scanpy's on purpose. scanpy defaults to
        Benjamini-Hochberg; this function uses Bonferroni unless told otherwise,
        because that is what the published results were corrected with and
        changing the default would silently move them.
    """

    if groupby not in adata.obs:
        raise KeyError(f"`groupby='{groupby}'` not found in adata.obs")

    # Decide which matrix to use
    if use_raw is None:
        use_raw = adata.raw is not None and layer is None

    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        Xmat = adata.raw.X
        var_names = np.asarray(adata.raw.var_names)
    else:
        Xmat = adata.layers[layer] if layer is not None else adata.X
        var_names = np.asarray(adata.var_names)

    # Determine groups
    cats = adata.obs[groupby]
    if hasattr(cats, "cat"):
        all_groups = list(cats.cat.categories)
    else:
        all_groups = sorted(map(str, np.unique(cats.values)))

    if groups is None or groups == "all":
        use_groups = all_groups.copy()
    else:
        use_groups = list(groups)

    if reference != "rest" and reference not in all_groups:
        raise ValueError(f"reference='{reference}' not found among {all_groups}")

    if reference != "rest":
        use_groups = [g for g in use_groups if g != reference]

    if len(use_groups) == 0:
        raise ValueError("No groups to test after applying `groups` and `reference`.")

    n_vars = len(var_names)
    store_all = n_genes is None
    n_store = n_vars if store_all else int(min(n_genes, n_vars))

    # Pre-allocate scanpy-like recarrays (but sized to n_store)
    dtype_names = [(g, object) for g in use_groups]
    dtype_float = [(g, "f4") for g in use_groups]

    names = np.recarray((n_store,), dtype=dtype_names)
    scores = np.recarray((n_store,), dtype=dtype_float)
    logfoldchanges = np.recarray((n_store,), dtype=dtype_float)
    pvals = np.recarray((n_store,), dtype=dtype_float)
    pvals_adj = np.recarray((n_store,), dtype=dtype_float)
    # Stored so a confidence interval is reachable without bypassing this
    # wrapper: lfc +/- 1.96 * lfc_se.
    lfc_se = np.recarray((n_store,), dtype=dtype_float)

    # Main loop
    obs_vals = adata.obs[groupby].values
    for g in use_groups:

        idx_g = obs_vals == g
        if reference == "rest":
            idx_r = ~idx_g
        else:
            idx_r = obs_vals == reference

        if idx_g.sum() == 0:
            raise ValueError(f"Group '{g}' has 0 cells.")
        if idx_r.sum() == 0:
            raise ValueError(f"Reference for group '{g}' has 0 cells (reference='{reference}').")

        Y_ = Xmat[idx_g, :]
        X_ = Xmat[idx_r, :]
        if sparse:
            lfc_vec, p_vec, statistic_vec = get_LN_lfcs_sparse(
                Y_,
                X_,
                test=test,
                return_statistic=True,
            )
        else:
            # This is slow: 0.1 seconds
            Y_ = _to_dense(Y_)
            X_ = _to_dense(X_)
            lfc_vec, p_vec, statistic_vec = get_LN_lfcs(
                Y_,
                X_,
                test=test,
                return_statistic=True,
            )
        lfc_vec = np.asarray(lfc_vec, dtype=float)
        p_vec = np.asarray(p_vec, dtype=float)
        statistic_vec = np.asarray(statistic_vec, dtype=float)
        # The standard error, without a second pass over the data.
        # statistic = (mu_Y - mu_X) / sqrt(se_Y^2 + se_X^2) and lfc = mu_Y - mu_X,
        # so sqrt(se_Y^2 + se_X^2) = lfc / statistic exactly. Calling the
        # estimator again for it would double the cost of every arm.
        with np.errstate(divide="ignore", invalid="ignore"):
            se_vec = np.abs(lfc_vec / statistic_vec)
        # lfc == 0 gives 0/0; the SE is finite there but unrecoverable this way,
        # so it is left as NaN rather than guessed at.
        se_vec[~np.isfinite(se_vec)] = np.nan
        q_vec = _adjust_pvalues(p_vec, corr_method, n_vars)

        safe_p = np.clip(p_vec, 1e-300, 1.0)
        # score_vec = np.sign(lfc_vec) * (-np.log10(safe_p))
        score_vec = statistic_vec

        # Rank genes
        if rankby_abs:
            order = np.argsort(-np.abs(score_vec))
        else:
            order = np.argsort(-score_vec)

        if store_all:
            top = order  # all genes, ranked
        else:
            top = order[:n_store]

        names[g] = var_names[top]
        scores[g] = score_vec[top].astype(np.float32)
        logfoldchanges[g] = lfc_vec[top].astype(np.float32)
        pvals[g] = p_vec[top].astype(np.float32)
        pvals_adj[g] = q_vec[top].astype(np.float32)
        lfc_se[g] = se_vec[top].astype(np.float32)

    # Write to adata.uns like scanpy
    adata.uns[key_added] = {
        "params": {
            "groupby": groupby,
            "groups": use_groups,
            "reference": reference,
            "n_genes": n_genes,
            "use_raw": bool(use_raw),
            "layer": layer,
            "method": "ln_wrapper",  # custom label
            "test": test,
            "rankby_abs": rankby_abs,
        },
        "names": names,
        "scores": scores,
        "logfoldchanges": logfoldchanges,
        "pvals": pvals,
        "pvals_adj": pvals_adj,
        "lfc_se": lfc_se,
    }

    return None
