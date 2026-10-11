import numpy as np
import scipy.sparse as sp

from ._ln_test import get_LN_lfcs, get_LN_lfcs_sparse, trigamma_diff


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


RANKBY = ("score", "markers")


def _detected(a):
    """Per gene, the number of cells with a positive value."""
    if sp.issparse(a):
        return np.asarray((a > 0).sum(axis=0)).ravel().astype(float)
    return (np.asarray(a) > 0).sum(axis=0).astype(float)


def _marker_scores(Y_, X_):
    """Per gene, the log ratio of the two detection rates over its standard error, detecting cells floored at 1."""
    n_y, n_x = Y_.shape[0], X_.shape[0]
    a_y = np.maximum(_detected(Y_), 1.0)
    a_x = np.maximum(_detected(X_), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((a_y / n_y) / (a_x / n_x)) / np.sqrt(trigamma_diff(a_y, n_y) + trigamma_diff(a_x, n_x))


def _marker_order(var_names, score_vec, q_vec, marker_vec, order):
    """The markers by marker score, then the other genes in ``order``.

    Markers and ties are judged on the float32 values the results are stored
    as, so the order is the one the stored statistics and p-values imply.
    """
    s = score_vec.astype(np.float32).astype(float)
    q = q_vec.astype(np.float32).astype(float)
    is_marker = (q < 0.05) & (s > 0)
    idx = np.flatnonzero(is_marker)
    m = marker_vec[idx]
    first = idx[np.lexsort((var_names[idx].astype(str), -s[idx], -np.where(np.isnan(m), -np.inf, m)))]
    return np.concatenate([first, order[~is_marker[order]]])


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
    corr_method: str = "benjamini-hochberg",
    rankby: str = "score",
):
    """
    Takes normalized data and performs LN's t-test. Updates the adata object with the results.

    corr_method
        Multiple-testing correction, using scanpy's vocabulary and scanpy's
        implementation: ``"benjamini-hochberg"`` or ``"bonferroni"``.
        The default is scanpy's, Benjamini-Hochberg.
    rankby
        ``"score"`` orders each group's genes by the test statistic, as scanpy
        does. ``"markers"`` puts the group's markers first: the genes with
        adjusted p below 0.05 and a positive statistic, ranked by the marker
        score, the log ratio of the group's and the reference's detection
        rates over its standard error (``sqrt(1/a - 1/n)`` per side, with ``a``
        the cells detecting the gene floored at 1), ties to the statistic and
        then the gene name; genes that every cell detects, scoring 0/0, come
        last among them. The other genes follow as ``"score"`` orders them,
        and the marker scores are stored as ``marker_scores``. The paper calls
        markers with ``corr_method="bonferroni"``.
    """

    if groupby not in adata.obs:
        raise KeyError(f"`groupby='{groupby}'` not found in adata.obs")
    if rankby not in RANKBY:
        raise ValueError(f"rankby must be one of {set(RANKBY)}.")

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
    if rankby == "markers":
        marker_scores = np.recarray((n_store,), dtype=dtype_float)

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
        if rankby == "markers":
            marker_vec = _marker_scores(Y_, X_)
            order = _marker_order(var_names, score_vec, q_vec, marker_vec, order)

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
        if rankby == "markers":
            marker_scores[g] = marker_vec[top].astype(np.float32)

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
            "rankby": rankby,
        },
        "names": names,
        "scores": scores,
        "logfoldchanges": logfoldchanges,
        "pvals": pvals,
        "pvals_adj": pvals_adj,
        "lfc_se": lfc_se,
    }
    if rankby == "markers":
        adata.uns[key_added]["marker_scores"] = marker_scores

    return None
