#!/usr/bin/env python3
"""Check that ``lntest`` reproduces ``utils_frozen``'s estimator on each arm.

Both use the same trigamma, so the two should agree up to ``utils_frozen``'s
float32 intermediates.

Two gates
---------
**Hard.** ``max|dLFC| < 1e-5`` on every measured arm. Anything larger means
something other than float32 differs, and the run is a failure, not a finding.

**Soft.** Significance-call disagreements at alpha = 0.05 should be zero; a
nonzero count means the estimators diverge somewhere. Reported, never failed on.

**If a headline number moves, stop.** That is a manuscript change and it goes to
Oskar. Do not soften, round, or average a difference away to get past this.

Exit codes
----------
``0`` both gates pass. ``2`` an arm was not measured, or was measured only in
part, so no verdict is available for it. ``3`` the hard gate failed. ``4`` a
headline number moved. ``5`` an API assertion failed. Nonzero is the normal
outcome of an incomplete run; it is never a reason to relax a gate.

Usage
-----
    python check_utils_vs_lntest.py --arm all --json report.json

Requires ``lntest`` importable (``pip install -e .`` at the repo root, or the
``de-ziln-reproducibility`` conda env). The comparison is against the
*installed* package, which is what every arm runs on.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from sklearn.metrics import confusion_matrix

# The frozen RECOMB-era estimator, imported as a plain sibling module: this file
# is run as a script, so reproducibility/ is sys.path[0] and no path hack is
# needed, and the reason both files are siblings rather than in equivalence/.
import utils_frozen

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The hard gate on the log-fold change. Not a tolerance to be relaxed: see the
# module docstring for why exceeding it is a failure rather than a finding.
HARD_DLFC_TOL = 1e-5

ALPHA = 0.05

# The five functions utils_frozen.py vendors, and which this harness re-verifies
# against utils.py for as long as utils.py exists.
FROZEN_FUNCS = ["trigamma", "get_DELN_lfcs", "get_LN_lfcs",
                "compute_p_vals", "get_t_statistic"]

# Callers of the utils API surface that has no lntest counterpart. These do NOT
# migrate: utils.py is kept as reproducibility/utils_frozen.py precisely so
# they can keep importing it.
#
# This constant was DELETION_LIST until 2026-09-10, when it named paths that
# were about to be deleted. Nothing is deleted any more -- the 3c re-judgement
# kept all 16 candidates -- so the assertion below changed premise with it. It
# used to read "every caller is deleted, so nothing needs translating"; it now
# reads "every caller stays on the frozen estimator, so nothing needs
# translating". Same conclusion, an honest reason, and one that survives a
# reader checking whether the files are actually gone.
STAYS_ON_FROZEN = (
    "reproducibility/lymphnode/gsea_utils.py",
    "reproducibility/lymphnode/ln_de_vs_rest.py",
    "reproducibility/lymphnode/cluster_de_gsea.py",
    "reproducibility/celltype/",
    "reproducibility/synthetic_nb/generate_lfc_data.py",
    "reproducibility/synthetic_nb/lfc_confidence_intervals.py",
)


# --------------------------------------------------------------------------
# Arm registry
# --------------------------------------------------------------------------

MEASURABLE = "measurable"
# These two strings are quoted verbatim in the vault; do not reword them.
NOT_APPLICABLE = "not applicable"
COVERED_BY_ARGUMENT = "covered by argument, not measured"


@dataclass
class Arm:
    letter: str
    name: str
    script: str
    status: str
    # Which utils entry point the arm actually calls. Arm C is the one that does
    # not call get_LN_lfcs, which is why it is an estimator substitution rather
    # than an import rename.
    utils_entry: str | None
    correction: str | None
    headline: str | None
    # Decimal places the manuscript prints this arm's headline metric to. The
    # "did it move" test is applied at this precision, and the raw delta is
    # reported regardless so the judgement is reviewable.
    headline_decimals: int | None
    note: str = ""


# NOTE: the letters A-H are used throughout the decision records but are never
# defined in one place. This mapping is inferred from the records that do name an
# arm (C = CITE-seq, F = lymph node, G = clustering, H = celltype) plus the
# handoff's "the two kidney experiments" and "the synthetic-NB experiments".
# It is the harness's own statement of scope and should be confirmed.
ARMS = [
    Arm("A", "synthetic NB null (FPR vs dispersion)", "variance_vs_metric.py",
        MEASURABLE, "get_LN_lfcs", "bonferroni", "fpr", 4,
        note="Pure null: mu1 == mu2, so every gene is a true negative. "
             "headline_decimals is 4, not the 2 the figure's axis suggests: FPR "
             "here is a count over 1500 genes, so its granularity is 1/1500 = "
             "6.7e-04 and every observed value is below 0.005. At 2 dp every "
             "entry rounds to 0.0 and `moved` could never fire, whatever the "
             "estimators did. 4 dp resolves a single significance call."),
    Arm("B", "synthetic NB with planted DEGs", "large_scale_NB_DE_test.py",
        MEASURABLE, "get_LN_lfcs", "bonferroni", "tpr,f1", 3,
        note="60 runs (3 dispersions x 20 reps). Headline is the per-dispersion "
             "mean, which is exactly the groupby cell large_scale_NB_latex_"
             "tables.py emits with float_format='%.3f' -- so the gate is 3 dp, "
             "matching the table, not the 2 this entry carried until the "
             "precision was checked against the generator."),
    Arm("C", "CITE-seq memory CD4", "large_scale_CITE_seq_exp.py",
        MEASURABLE, "get_DELN_lfcs", "fdr_bh", "tpr,f1", 3,
        note="Estimator substitution, not an import rename: calls get_DELN_lfcs, "
             "which has no lntest counterpart. Its eps=1e-9 term and missing "
             "all-zero guard must be shown to be inert behind the arm's own "
             "min_cells_per_group=3 filter, not assumed to be."),
    Arm("D", "Visium HD kidney, UMI subsampling",
        "notebooks/test/vishd_test_parallel.py", COVERED_BY_ARGUMENT,
        "get_LN_lfcs", None, None, None,
        note="Aliases get_LN_lfcs as get_DELN_lfcs -- the same CODE PATH arms "
             "A and B measure, in a REGIME THEY DO NOT REACH. A and B's zero "
             "disagreements came from detection counts of ~980 and ~7818, "
             "where the two trigammas agreed to ~1e-3, while this arm "
             "binomially downsamples UMIs so a_hat reaches 1-2, where they "
             "differed by 22-44%. RESOLVED 2026-09-11: there is one "
             "trigamma now and no substitution to measure, in this "
             "regime or any other. The arm still has not been run -- the "
             "download has not happened -- but it can no longer move a "
             "published number through this term. "
             "Note vishd_test_de_parallel.py carries the identical alias and "
             "is not assigned a letter here."),
    Arm("E", "Visium HD kidney, spot subsampling",
        "notebooks/test/vishd_test_shape_split_shared.py", COVERED_BY_ARGUMENT,
        "get_LN_lfcs", None, None, None,
        note="Same alias and same missing download as arm D. Reaches the "
             "regime differently: shape_id aggregation makes each row a "
             "pseudo-bulk, so entries are denser but n is the number of "
             "capsules -- tens -- and small n also put 1/a - 1/n far from "
             "sum 1/j^2. RESOLVED on the same grounds as arm D: one trigamma, "
             "no substitution, so the regime no longer matters."),
    Arm("F", "lymph node subsampling",
        "reproducibility/run_subsampling_analysis_parallel.py", NOT_APPLICABLE,
        None, None, None, None,
        note="Reaches the estimator through de_utils.run_ln_de -> "
             "rank_genes_groups_ln -> get_LN_lfcs_sparse and never imports "
             "utils. This is a proof of no-effect, not an untested gap."),
    Arm("G", "PBMC3k clustering",
        "reproducibility/clustering/run_clustering_de.py", NOT_APPLICABLE,
        None, None, None, None,
        note="Same proof as arm F. Run anyway for continuity with "
             "the dense/sparse reconciliation, and because it is the only arm "
             "with committed reference checksums."),
]


# --------------------------------------------------------------------------
# Self-checks: the harness must not be trusted more than it has earned
# --------------------------------------------------------------------------

def _extract_funcs(path: pathlib.Path, names):
    src = path.read_text()
    tree, lines = ast.parse(src), src.splitlines(keepends=True)
    return {n.name: "".join(lines[n.lineno - 1:n.end_lineno]).rstrip()
            for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


def verify_frozen_copy():
    """Confirm utils_frozen.py is still a verbatim copy of utils.py.

    After stage 3d deletes utils.py this can no longer be checked, and saying so
    is the point: a referee running this on the post-refactor tree should be told
    that the frozen copy is now the only witness, not left to assume it was
    verified.
    """
    utils = REPO_ROOT / "utils.py"
    frozen = pathlib.Path(__file__).with_name("utils_frozen.py")
    if not utils.exists():
        return ("unverifiable", "utils.py is gone (expected after stage 3d); "
                                "utils_frozen.py is now the only copy of the "
                                "RECOMB-era estimator and cannot be re-checked.")
    a, b = _extract_funcs(utils, FROZEN_FUNCS), _extract_funcs(frozen, FROZEN_FUNCS)
    drifted = [n for n in FROZEN_FUNCS if a.get(n) != b.get(n)]
    if drifted:
        raise SystemExit(
            f"FATAL: utils_frozen.py has drifted from utils.py in {drifted}. "
            "The frozen copy must be verbatim or every number below is measured "
            "against the wrong estimator."
        )
    digest = hashlib.sha256(utils.read_bytes()).hexdigest()
    return ("verified", f"all {len(FROZEN_FUNCS)} functions verbatim against "
                        f"utils.py sha256 {digest[:16]}")


# --------------------------------------------------------------------------
# The arms' own evaluation metric, vendored
# --------------------------------------------------------------------------

# Verbatim from scanpy_ttest.py:63-93, vendored for the same reason
# utils_frozen.py exists. Importing it instead would pull scanpy and utils.py in
# from the repository root, which needs the sys.path hack this refactor removes,
# and stage 3d deletes that file anyway -- get_test_results moves to
# reproducibility/baselines.py. Recomputing each arm's headline metric is the
# whole point of this harness, so the metric has to outlive the move.
# verify_frozen_metric() re-checks this copy against whichever file is present.
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


METRIC_FUNCS = ["get_test_results"]

# Where get_test_results lives, before and after stage 3d. Checked in order.
METRIC_SOURCES = ("reproducibility/baselines.py", "scanpy_ttest.py")


def verify_frozen_metric():
    """Confirm the vendored get_test_results still matches the arms' own copy."""
    mine = _extract_funcs(pathlib.Path(__file__), METRIC_FUNCS)
    for rel in METRIC_SOURCES:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        theirs = _extract_funcs(src, METRIC_FUNCS)
        if theirs.get("get_test_results") != mine.get("get_test_results"):
            raise SystemExit(
                f"FATAL: the vendored get_test_results has drifted from {rel}. "
                "Every headline metric below would then be computed by a "
                "different rule than the arm itself computes it by."
            )
        return ("verified", f"get_test_results verbatim against {rel}")
    return ("unverifiable",
            "neither " + " nor ".join(METRIC_SOURCES) + " is present; the "
            "vendored get_test_results is now the only copy and cannot be "
            "re-checked.")


def _rng_digest():
    """A comparable fingerprint of numpy's global RNG state."""
    state = np.random.get_state()
    h = hashlib.sha256()
    h.update(str(state[0]).encode())
    h.update(np.asarray(state[1]).tobytes())
    h.update(str(state[2:]).encode())
    return h.hexdigest()


def check_baseline_rng_neutrality():
    """The precondition that lets the adapters run only the LN path.

    Every arm draws its inputs from numpy's *global* RNG and interleaves its
    scanpy baselines with the draws for the following replicate. Skipping those
    baselines -- which the substitution cannot affect, since neither estimator
    is involved in them -- therefore reproduces the arm's inputs only if scanpy
    consumes no randomness of its own. If it does, every replicate after the
    first is measured on data the published run never saw, and every number this
    harness reports is measured on the wrong inputs.

    This exercises the four scanpy entry points scanpy_sig_test calls rather
    than the wrapper, because the wrapper is at the repository root today and a
    sibling baselines.py after stage 3d, and importing it either way needs the
    sys.path hack this refactor exists to remove. The rest of scanpy_sig_test is
    array reshaping with no RNG in it, and under CP10K -- which is the branch
    all three arms take -- it does not mutate its inputs either
    (scanpy_ttest.py:37-48); the median-of-ratios branch does, and no arm here
    reaches it.

    A tiny input is not a proof for a large one: a method that subsampled only
    above some size would slip through. It is the strongest cheap check there
    is, and it is reported as a check rather than as a guarantee.
    """
    label = "the scanpy baselines consume no numpy randomness"
    saved = np.random.get_state()
    try:
        import scanpy as sc
    except ImportError as exc:
        return {"check": label, "ok": None,
                "detail": f"scanpy is not importable ({exc}), so this was not "
                          "checked. Arms A, B and C assume it."}
    try:
        np.random.seed(0)
        X = np.random.negative_binomial(n=1.0, p=0.5, size=(20, 12))
        Y = np.random.negative_binomial(n=1.0, p=0.5, size=(20, 12))
        before = _rng_digest()
        adata = sc.AnnData(np.vstack([X, Y]).astype(float))
        adata.var_names = [f"Gene{i}" for i in range(12)]
        adata.obs["group"] = np.concatenate([np.repeat("X", 20), np.repeat("Y", 20)])
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        for method in ("t-test", "wilcoxon"):
            sc.tl.rank_genes_groups(adata, groupby="group", method=method,
                                    reference="X", corr_method="bonferroni")
        after = _rng_digest()
    except Exception as exc:  # a check that cannot run says so; it does not pass
        return {"check": label, "ok": None,
                "detail": f"the check itself failed to run ({exc!r}); unverified."}
    finally:
        np.random.set_state(saved)

    if before != after:
        raise SystemExit(
            "FATAL: scanpy moved numpy's global RNG state. The arm adapters "
            "skip the baselines, so every replicate after the first would be "
            "measured on inputs the published run never saw. The adapters have "
            "to run the baselines too before any number below means anything."
        )
    return {"check": label, "ok": True,
            "detail": "sc.AnnData, normalize_total, log1p and rank_genes_groups "
                      "(t-test and wilcoxon) left the global RNG state "
                      "unchanged on a 40x12 input."}


def require_lntest():
    """Return the installed package's estimator module.

    ``lntest`` itself is the import that must succeed -- the whole point is to
    measure against the *installed* package rather than against ``pkg/`` on a
    path. Everything downstream is given the ``lntest._ln_test`` submodule
    specifically, which keeps the get_DELN_lfcs assertion below meaningful:
    resolved against a package that re-exports names, it could pass for the
    wrong reason.

    ``lntest.get_LN_lfcs`` and ``lntest.rank_genes_groups_ln`` do now exist as
    top-level names (added 2026-09-10), and since the module rename they are
    the *only* public route: both implementation modules are private.
    """
    try:
        import lntest
        import lntest._ln_test as estimator
    except ImportError as exc:
        raise SystemExit(
            f"FATAL: cannot import lntest ({exc}).\n"
            "This harness compares against the published package on purpose.\n"
            "Install it with `pip install -e .` at the repo root, or activate "
            "the de-ziln-reproducibility conda env."
        )
    return estimator


def _signature_params(fn):
    import inspect
    return inspect.signature(fn).parameters


def check_api_mismatches(lntest):
    """Assert the three API mismatches rather than trusting a grep.

    These are the rewiring's real work. Each is
    asserted here so that a change to either module fails this harness rather
    than surfacing as a silent behaviour change during stage 3b.
    """
    findings = []

    # (1) return_log_abs_statistic. This was "has no lntest counterpart, and every
    #     caller must be on the deletion list". Both halves are now out of date:
    #     the deletion list is void (the 3c re-judgement kept everything) and the
    #     package gained the parameter on 2026-09-10. So the assertion inverts --
    #     it now checks the counterpart EXISTS and returns the log of its own
    #     statistic, which is the property the GSEA ranking depends on.
    rng = np.random.default_rng(0)
    tiny_Y = rng.integers(1, 20, size=(40, 12)).astype(float)
    tiny_X = rng.integers(1, 20, size=(40, 12)).astype(float)
    has_param = "return_log_abs_statistic" in _signature_params(lntest.get_LN_lfcs)
    consistent = None
    if has_param:
        _lfc, _p, st, la = lntest.get_LN_lfcs(tiny_Y, tiny_X, test='t',
                                              return_log_abs_statistic=True)
        ok_mask = np.isfinite(st) & (np.abs(st) > 0)
        consistent = float(np.max(np.abs(la[ok_mask] - np.log(np.abs(st[ok_mask])))))
    findings.append({
        "mismatch": "return_log_abs_statistic has an lntest counterpart",
        "present": has_param,
        "max_abs_deviation_from_log_abs_statistic": consistent,
        "ok": has_param and consistent is not None and consistent < 1e-12,
        "detail": "lntest.get_LN_lfcs(return_log_abs_statistic=True) returns "
                  "(lfc, p_vals, statistic, log_abs) -- note the order differs "
                  "from utils, which returned (lfc, statistic, log_abs, p_vals). "
                  "Callers translate rather than repoint."
                  if has_param else
                  "MISSING: lymphnode/gsea_utils.py and ln_de_vs_rest.py need it.",
    })

    # (2) The return flags are mutually exclusive. Until 2026-09-10 lntest
    #     silently dropped the statistic when both were set, and this assertion
    #     pinned that defect while the vault recorded that a guard
    #     was owed. The guard now exists, so the assertion checks it fires --
    #     which is why this harness failed loudly the moment the API changed
    #     underneath it, rather than passing on a stale premise.
    try:
        lntest.get_LN_lfcs(tiny_Y, tiny_X, return_standard_error=True,
                           return_statistic=True)
        raised = False
    except ValueError:
        raised = True
    findings.append({
        "mismatch": "the return flags are mutually exclusive and say so",
        "raises_when_combined": raised,
        "ok": raised,
        "detail": "A caller asking for two quantities now gets an error rather "
                  "than one of them."
                  if raised else
                  "REGRESSION: the guard is gone; the statistic is being dropped "
                  "silently again.",
    })

    # (3) get_DELN_lfcs has no counterpart at all -- this is what makes arm C a
    #     substitution rather than a rename.
    findings.append({
        "mismatch": "get_DELN_lfcs has no lntest counterpart",
        "ok": not hasattr(lntest, "get_DELN_lfcs"),
        "detail": "Arm C must be measured, not assumed equivalent.",
    })
    return findings


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------

def compare_estimators(lfc_u, p_u, lfc_l, p_l, correction, alpha=ALPHA):
    """Compare one paired run of the two estimators on identical inputs.

    ``_u`` is utils_frozen, ``_l`` is lntest. Both must have been given exactly
    the same X and Y; producing that is each arm adapter's job, and is the part
    of this harness most able to be silently wrong.
    """
    import statsmodels.stats.multitest as smm

    lfc_u, p_u = np.asarray(lfc_u, float), np.asarray(p_u, float)
    lfc_l, p_l = np.asarray(lfc_l, float), np.asarray(p_l, float)

    dlfc = np.abs(lfc_u - lfc_l)
    finite_dlfc = dlfc[np.isfinite(dlfc)]

    # p can be exactly 0, so log10 is only defined on part of the vector. Report
    # how many genes were excluded rather than flooring them into the statistic.
    both_pos = (p_u > 0) & (p_l > 0) & np.isfinite(p_u) & np.isfinite(p_l)
    dlog10p = np.abs(np.log10(p_u[both_pos]) - np.log10(p_l[both_pos]))

    # Each arm applies its own correction; significance is judged after it.
    def adjust(p):
        ok = np.isfinite(p)
        out = np.full_like(p, np.nan)
        if ok.any():
            out[ok] = smm.multipletests(p[ok], alpha=alpha, method=correction)[1]
        return out

    adj_u, adj_l = adjust(p_u), adjust(p_l)
    sig_u, sig_l = adj_u < alpha, adj_l < alpha

    return {
        "n_genes": int(lfc_u.size),
        "max_abs_dlfc": float(finite_dlfc.max()) if finite_dlfc.size else None,
        "median_abs_dlfc": float(np.median(finite_dlfc)) if finite_dlfc.size else None,
        "max_abs_dlog10p": float(dlog10p.max()) if dlog10p.size else None,
        "n_excluded_from_dlog10p": int((~both_pos).sum()),
        # Split by direction: which estimator called the gene significant.
        "disagree_utils_only": int((sig_u & ~sig_l).sum()),
        "disagree_lntest_only": int((sig_l & ~sig_u).sum()),
        "disagree_total": int((sig_u != sig_l).sum()),
        "nan_lfc_utils": int(np.isnan(lfc_u).sum()),
        "nan_lfc_lntest": int(np.isnan(lfc_l).sum()),
        "nan_p_utils": int(np.isnan(p_u).sum()),
        "nan_p_lntest": int(np.isnan(p_l).sum()),
    }


def headline_moved(value_u, value_l, decimals):
    """Did the published number change at the precision the manuscript prints?

    Reported alongside the raw delta, never instead of it.
    """
    return round(float(value_u), decimals) != round(float(value_l), decimals)


# --------------------------------------------------------------------------
# Arm adapters -- stage 1b-ii
# --------------------------------------------------------------------------
#
# Each adapter transcribes one arm script's input generation line for line: the
# same seed, the same draw order, the same filtering. The scripts cannot be
# imported -- they are top-level code that calls plt.show() and writes to
# hardcoded paths on other people's machines -- so transcription is the only way
# to put both estimators on an arm's real inputs, and it is the single place
# this harness is most able to be quietly wrong. Read each adapter against its
# script; that is what this commit is for.
#
# Four things are deliberately *not* transcribed.
#
#   * The scanpy and wilcoxon baselines, which neither estimator touches. That
#     is safe only if they consume no numpy randomness, because every arm
#     interleaves them with the next replicate's draws --
#     check_baseline_rng_neutrality() is that precondition, checked rather than
#     assumed, and fatal if it fails.
#   * The plotting, the per-replicate CSV dumps and the results CSVs, none of
#     which feed a number.
#   * Arm C's per-replicate AnnData copy and densification, which are identical
#     in every replicate and consume no randomness. Nothing in the loop writes
#     to the dense matrix, so they are hoisted out of it.
#   * Arm B's `method` loop, which rebinds `method` to "Scanpy t-test" inside
#     itself. Only its LN branch is transcribed.
#
# --reps evaluates a prefix of the replicates but *always performs every draw*,
# so a reduced run sees exactly the inputs the full run's first replicates see
# rather than a differently-seeded stream. Arm B is the exception worth naming:
# its replicate loop is nested inside a dispersion loop, so a truncated run is a
# true prefix of the full run only within each dispersion block, and the blocks
# themselves stay aligned only because the skipped work is evaluation, not
# drawing. Reduced runs are labelled, and their headline numbers are marked not
# comparable to the manuscript.


class MissingInput(RuntimeError):
    """An arm's input is not on disk. Not a gate failure -- an unmeasured arm."""


def _pair_LN(Y, X, lntest):
    """The utils entry arms A and B call, against its lntest counterpart."""
    lfc_u, p_u = utils_frozen.get_LN_lfcs(Y, X, test='t')
    lfc_l, p_l = lntest.get_LN_lfcs(Y, X, test='t')
    return lfc_u, p_u, lfc_l, p_l


def _abs_delta(a, b):
    return np.abs(np.asarray(a, float) - np.asarray(b, float))


def _headline(metric, value_u, value_l, decimals):
    """One headline number, both ways, with the raw delta kept alongside."""
    value_u, value_l = float(value_u), float(value_l)
    return {
        "metric": metric,
        "utils": value_u,
        "lntest": value_l,
        "delta": value_l - value_u,
        "decimals": decimals,
        "utils_rounded": round(value_u, decimals),
        "lntest_rounded": round(value_l, decimals),
        "moved": headline_moved(value_u, value_l, decimals),
    }


def _aggregate(units, dlfc_pool):
    """Roll the per-replicate comparisons up into the arm's row of the record.

    max and median of |dLFC| are pooled over every gene of every replicate.
    We want the arm's max|dLFC| and median|dLFC|, and
    a max of per-replicate medians is neither of those.
    """
    def total(key):
        return int(sum(u[key] for u in units))

    pool = np.concatenate(dlfc_pool) if dlfc_pool else np.empty(0)
    finite = pool[np.isfinite(pool)]
    dlog10p = [u["max_abs_dlog10p"] for u in units if u["max_abs_dlog10p"] is not None]
    return {
        "n_units": len(units),
        "n_gene_tests": int(pool.size),
        "max_abs_dlfc": float(finite.max()) if finite.size else None,
        "median_abs_dlfc": float(np.median(finite)) if finite.size else None,
        "max_abs_dlog10p": max(dlog10p) if dlog10p else None,
        "n_excluded_from_dlog10p": total("n_excluded_from_dlog10p"),
        "disagree_utils_only": total("disagree_utils_only"),
        "disagree_lntest_only": total("disagree_lntest_only"),
        "disagree_total": total("disagree_total"),
        "nan_lfc_utils": total("nan_lfc_utils"),
        "nan_lfc_lntest": total("nan_lfc_lntest"),
        "nan_p_utils": total("nan_p_utils"),
        "nan_p_lntest": total("nan_p_lntest"),
    }


def _finish(units, dlfc_pool, headline, provenance, complete):
    agg = _aggregate(units, dlfc_pool)
    max_dlfc = agg["max_abs_dlfc"]
    return {
        "provenance": provenance,
        "complete": complete,
        "headline_comparable_to_manuscript": complete,
        "aggregate": agg,
        "headline": headline,
        "headline_moved": any(h["moved"] for h in headline),
        "hard_gate": {
            "max_abs_dlfc": max_dlfc,
            "tol": HARD_DLFC_TOL,
            # Nothing measured is not a pass.
            "ok": max_dlfc is not None and max_dlfc < HARD_DLFC_TOL,
        },
        "units": units,
    }


# --------------------------------------------------------------------------

def _arm_a_metric(p_vals, true_lfcs, n_genes):
    """variance_vs_metric.py:50-55, its 1x1-confusion-matrix guard included."""
    adj = smm.multipletests(p_vals, alpha=0.05, method='bonferroni')[1]
    if np.sum(adj >= 0.05) == n_genes:
        # 100% accuracy and 0% TPR --- conf_mat crashes in this setting
        return {"accuracy": 1., "fpr": 0.}
    return get_test_results(adj, true_lfcs, verbose=False)


def _arm_a(arm, lntest, reps=None):
    """variance_vs_metric.py -- FPR on a pure null over a 4 x 20 dispersion grid.

    Transcribed from variance_vs_metric.py:11-58. Seed 0; nx = ny = 1000 cells,
    1500 genes, base_mu = 50 in *both* groups, so true_lfcs is all zero, every
    gene is a true negative and FPR is the whole of the arm. X is drawn before Y
    in each grid cell, and the estimator is then called as get_LN_lfcs(Y, X) --
    the reverse of the draw order, which is easy to lose in transcription.

    The arm's own guard is transcribed too. On a pure null the test frequently
    calls nothing significant, sklearn's confusion matrix is then 1x1 and
    crashes, and the script substitutes an FPR of 0 rather than computing one.
    That branch is the arm's definition of its metric, not a workaround.

    The manuscript figure is a curve, not a scalar, so every grid cell is a
    headline number here and fpr_grid_mean is a summary of them rather than a
    replacement for them.
    """
    np.random.seed(0)
    nx = 1000
    ny = 1000
    n_genes = 1500
    base_mu = 50
    mu1 = base_mu + np.zeros((1, n_genes))
    mu2 = base_mu
    true_lfcs = np.log2(mu2 / mu1)

    d1_list = np.linspace(0.01, 1, 4)
    d2_list = np.linspace(0.0001, 2, 20)
    var_1 = d1_list * base_mu ** 2 + base_mu
    var_2 = d2_list * base_mu ** 2 + base_mu
    metric = "fpr"

    n_cells = d1_list.size * d2_list.size
    limit = n_cells if reps is None else max(0, min(int(reps), n_cells))

    units, dlfc_pool, headline = [], [], []
    fpr_u_all, fpr_l_all = [], []
    cell = 0
    for i, d1 in enumerate(d1_list):
        for j, d2 in enumerate(d2_list):
            r1 = 1 / d1
            r2 = 1 / d2
            p1 = 1 / (1 + d1 * mu1)
            p2 = 1 / (1 + d2 * mu2)

            X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
            Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))
            # the two baselines the script runs here are skipped; see above

            if cell < limit:
                lfc_u, p_u, lfc_l, p_l = _pair_LN(Y, X, lntest)
                unit = compare_estimators(lfc_u, p_u, lfc_l, p_l, arm.correction)
                unit["unit"] = {"d1": float(d1), "d2": float(d2),
                                "var_X": int(var_1[i]), "var_Y": int(var_2[j])}
                units.append(unit)
                dlfc_pool.append(_abs_delta(lfc_u, lfc_l))

                f_u = _arm_a_metric(p_u, true_lfcs, n_genes)[metric]
                f_l = _arm_a_metric(p_l, true_lfcs, n_genes)[metric]
                fpr_u_all.append(f_u)
                fpr_l_all.append(f_l)
                headline.append(_headline(
                    f"fpr[Var(X)={int(var_1[i])},Var(Y)={int(var_2[j])}]",
                    f_u, f_l, arm.headline_decimals))
            cell += 1

    if fpr_u_all:
        headline.append(_headline("fpr_grid_mean", np.mean(fpr_u_all),
                                  np.mean(fpr_l_all), arm.headline_decimals))

    provenance = {
        "script": arm.script, "seed": 0, "nx": nx, "ny": ny, "n_genes": n_genes,
        "base_mu": base_mu, "metric": metric,
        "d1_list": [float(v) for v in d1_list],
        "d2_list": [float(v) for v in d2_list],
        "cells_evaluated": len(units), "cells_total": int(n_cells),
        "utils_entry": "get_LN_lfcs",
        "note": "Pure null: mu1 == mu2 for every gene, so the only positives "
                "either estimator can produce are false ones.",
    }
    return _finish(units, dlfc_pool, headline, provenance, len(units) == n_cells)


# --------------------------------------------------------------------------

def _arm_b_metrics(p_vals, z1):
    """large_scale_NB_DE_test.py:49-55, LN branch only."""
    adj = smm.multipletests(p_vals, alpha=0.05, method='bonferroni')[1]
    r = get_test_results(adj, z1, verbose=False)
    return r["tpr"], r["f1"]


def _arm_b(arm, lntest, reps=None):
    """large_scale_NB_DE_test.py -- planted DEGs, 3 dispersions x 20 replicates.

    Transcribed from large_scale_NB_DE_test.py:8-59. Seed 0; nx = ny = 10000
    cells, 1500 genes, non_de_mu = 10, d2 fixed at 1, and a batch factor of e
    applied to the first half of each group's cells and then *not* removed
    before drawing. Per replicate the draw order is z1, then the Gaussian effect
    sizes -- drawn whether or not z1 selected the gene, so the stream does not
    depend on z1 -- then X, then Y. The estimator is called as get_LN_lfcs(Y, X).

    The ground truth handed to get_test_results is z1 itself, not the realised
    log-fold change, so a gene z1 selected whose Gaussian draw came out near
    zero still counts as a true positive. That is the arm's definition and it is
    transcribed, not corrected.

    Peak memory is several GB per replicate: mu1, p1, X and Y are each
    10000 x 1500, and both estimators copy their inputs to float64.
    """
    np.random.seed(0)
    nx = 10000
    ny = 10000
    n_genes = 1500
    d2 = 1.
    non_de_mu = 10
    log_batch_factor = 1.
    rep_count = 20
    dispersions = [1., 1.5, 2.]
    limit = rep_count if reps is None else max(0, min(int(reps), rep_count))

    units, dlfc_pool, headline = [], [], []
    tpr_u_all, tpr_l_all, f1_u_all, f1_l_all = [], [], [], []

    for d1 in dispersions:
        tpr_u, tpr_l, f1_u, f1_l = [], [], [], []
        for rep in range(rep_count):
            z1 = np.random.binomial(1, 0.1, (1, n_genes))
            if non_de_mu == 10:
                mu1 = non_de_mu + np.abs(np.random.normal(0, 5, (1, n_genes))) * z1
            else:
                mu1 = non_de_mu + np.random.normal(15, 5, (1, n_genes)) * z1
            mu1 = np.tile(mu1, (nx, 1))
            mu2 = non_de_mu * np.ones((ny, n_genes))

            # the first half of all samples are batch effected
            mu1[:nx // 2] *= np.exp(log_batch_factor)
            mu2[:ny // 2] *= np.exp(log_batch_factor)

            r1 = 1 / d1
            r2 = 1 / d2
            p1 = 1 / (1 + d1 * mu1)
            p2 = 1 / (1 + d2 * mu2)

            X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
            Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))
            # the script's t-test and wilcoxon branches are skipped; see above

            if rep >= limit:
                continue

            lfc_u, p_u, lfc_l, p_l = _pair_LN(Y, X, lntest)
            unit = compare_estimators(lfc_u, p_u, lfc_l, p_l, arm.correction)
            unit["unit"] = {"dispersion": float(d1), "rep_no": rep}
            units.append(unit)
            dlfc_pool.append(_abs_delta(lfc_u, lfc_l))

            u_tpr, u_f1 = _arm_b_metrics(p_u, z1)
            l_tpr, l_f1 = _arm_b_metrics(p_l, z1)
            tpr_u.append(u_tpr)
            f1_u.append(u_f1)
            tpr_l.append(l_tpr)
            f1_l.append(l_f1)

        if tpr_u:
            headline.append(_headline(f"tpr[dispersion={d1}]", np.mean(tpr_u),
                                      np.mean(tpr_l), arm.headline_decimals))
            headline.append(_headline(f"f1[dispersion={d1}]", np.mean(f1_u),
                                      np.mean(f1_l), arm.headline_decimals))
        tpr_u_all += tpr_u
        tpr_l_all += tpr_l
        f1_u_all += f1_u
        f1_l_all += f1_l

    if tpr_u_all:
        headline.append(_headline("tpr_mean_all", np.mean(tpr_u_all),
                                  np.mean(tpr_l_all), arm.headline_decimals))
        headline.append(_headline("f1_mean_all", np.mean(f1_u_all),
                                  np.mean(f1_l_all), arm.headline_decimals))

    provenance = {
        "script": arm.script, "seed": 0, "nx": nx, "ny": ny, "n_genes": n_genes,
        "non_de_mu": non_de_mu, "d2": d2, "log_batch_factor": log_batch_factor,
        "dispersions": dispersions, "rep_count": rep_count,
        "reps_evaluated_per_dispersion": limit,
        "utils_entry": "get_LN_lfcs",
        "ground_truth": "z1, the Bernoulli(0.1) selection indicator, not the "
                        "realised log-fold change",
        "note": "The batch factor is applied to mu before drawing and never "
                "removed, unlike scanpy_ttest.py's __main__ block, which "
                "removes it again to build its ground truth.",
    }
    return _finish(units, dlfc_pool, headline, provenance,
                   limit == rep_count)


# --------------------------------------------------------------------------

def _arm_c_metrics(p_vals, true_lfcs, lfc, correction):
    """large_scale_CITE_seq_exp.py:100, 125 and 129."""
    adj = smm.multipletests(p_vals, alpha=0.05, method=correction)[1]
    r = get_test_results(adj, true_lfcs, verbose=False)
    return r["tpr"], r["f1"], float(np.mean((lfc - true_lfcs) ** 2))


def _arm_c(arm, lntest, reps=None):
    """large_scale_CITE_seq_exp.py -- planted signal on real CITE-seq counts.

    Transcribed from large_scale_CITE_seq_exp.py:29-135. Seed 1, 100 replicates
    over citeseq/data/memory_CD4.h5ad with the MT- genes dropped. Per
    replicate the draw order is the group assignment over all cells, then the
    100 Gaussian log-fold changes, then the 100 gene indices they are applied to.

    Two literal details that would leave the magnitudes right and the sign or the
    filter wrong if lost. This arm calls **get_DELN_lfcs, not get_LN_lfcs**, and
    it calls it as (X_data_filtered, Y_data_filtered) -- so the perturbed
    group_B is the estimator's first argument and the log-fold change runs
    group_B minus group_A. And the detection filter is computed on the
    *unperturbed* group_B counts, X_data_orig, while the estimator is given the
    perturbed X_data_modified.

    The eps = 1e-9 term and the missing
    all-zero-gene guard that separate get_DELN_lfcs from get_LN_lfcs to be
    *shown* inert behind this arm's min_cells_per_group = 3 filter rather than
    assumed inert. So each replicate also runs utils' own get_LN_lfcs on the same
    inputs, with the trigamma held fixed, reported as deln_vs_ln_within_utils
    next to a count of all-zero genes surviving the filter.

    That gap is NOT the eps and the guard, which is what this docstring claimed
    until the numbers came in. Both are provably inert -- the filter guarantees
    n_plus >= 3, so eps ** (1 + n_plus) <= 1e-36, and 0 all-zero genes survive,
    so the guard branch is unreachable. The measured 1.03e-06 is the float32
    intermediates get_LN_lfcs allocates and get_DELN_lfcs does not.
    """
    try:
        import anndata as ann
    except ImportError as exc:
        raise MissingInput(
            f"anndata is not importable ({exc}); arm C needs it to read "
            "memory_CD4.h5ad"
        ) from None

    path = pathlib.Path(__file__).resolve().parent / "citeseq" / "data" / "memory_CD4.h5ad"
    if not path.exists():
        raise MissingInput(
            f"{path} is missing. It is a derived intermediate, rebuildable from "
            "the public 10x matrix by citeseq/R/pbmc10k_process.R and "
            "pbmc10k_to_h5ad.R, and it is one of the files "
            "the Zenodo deposit has it."
        )
    memory_CD4 = ann.read_h5ad(path)

    np.random.seed(1)
    replicates = 100
    min_cells_per_group = 3
    target_group = 'group_B'
    n_signal_genes = 100
    limit = replicates if reps is None else max(0, min(int(reps), replicates))

    # Hoisted out of the replicate loop: the script rebuilds this from a fresh
    # copy every replicate, it consumes no randomness, and nothing in the loop
    # writes to it -- X_data_modified is a copy of a fancy-indexed copy.
    mt_gene_mask = memory_CD4.var_names.str.startswith('MT-')
    adata = memory_CD4[:, ~mt_gene_mask].copy()
    var_names = adata.var_names
    n_obs, n_genes = adata.n_obs, adata.n_vars
    adata_dense = adata.X.toarray()

    units, dlfc_pool, headline = [], [], []
    tpr_u, tpr_l, f1_u, f1_l, mse_u, mse_l = [], [], [], [], [], []
    within_dlfc, within_dp, all_zero_after_filter = [], [], []

    for i in range(replicates):
        group_assignments = np.random.choice(['group_A', 'group_B'], size=n_obs)
        log2_fold_change = np.random.normal(loc=0.0, scale=1.0, size=n_signal_genes)
        fold_change = np.power(2, log2_fold_change)
        gene_indices_to_modify = np.random.choice(n_genes, size=n_signal_genes,
                                                  replace=False)
        if i >= limit:
            continue

        signal_gene_names = var_names[gene_indices_to_modify]
        # the script sets this on adata_rep.var and reads it back, which routes
        # the indices through the gene names; kept as such rather than shortcut
        # to a positional mask
        genes_to_modify_mask = pd.Series(False, index=var_names)
        genes_to_modify_mask.loc[signal_gene_names] = True

        true_lfcs = np.zeros(n_genes)
        true_lfcs[genes_to_modify_mask] = log2_fold_change

        cells_to_modify_mask = group_assignments == target_group
        Y_data = adata_dense[~cells_to_modify_mask, :]
        X_data_orig = adata_dense[cells_to_modify_mask, :]
        X_data_modified = X_data_orig.copy()
        X_data_modified[:, genes_to_modify_mask] *= fold_change

        # both filters are computed on the unperturbed counts
        expr_in_A = np.sum(Y_data > 0, axis=0) >= min_cells_per_group
        expr_in_B = np.sum(X_data_orig > 0, axis=0) >= min_cells_per_group
        gene_filter_mask = expr_in_A & expr_in_B

        X_data_filtered = X_data_modified[:, gene_filter_mask]
        Y_data_filtered = Y_data[:, gene_filter_mask]
        true_lfcs_filtered = true_lfcs[gene_filter_mask]

        lfc_u, p_u, _ = utils_frozen.get_DELN_lfcs(
            X_data_filtered, Y_data_filtered, return_standard_error=True)
        lfc_l, p_l, _ = lntest.get_LN_lfcs(
            X_data_filtered, Y_data_filtered, return_standard_error=True)
        # the same inputs through utils' *other* function, which isolates the
        # eps and the all-zero guard with the trigamma held fixed
        lfc_ln_u, p_ln_u, _ = utils_frozen.get_LN_lfcs(
            X_data_filtered, Y_data_filtered, return_standard_error=True)

        unit = compare_estimators(lfc_u, p_u, lfc_l, p_l, arm.correction)
        unit["unit"] = {"replicate": i,
                        "n_genes_after_filter": int(gene_filter_mask.sum())}
        units.append(unit)
        dlfc_pool.append(_abs_delta(lfc_u, lfc_l))

        within_dlfc.append(float(np.nanmax(_abs_delta(lfc_u, lfc_ln_u))))
        within_dp.append(float(np.nanmax(_abs_delta(p_u, p_ln_u))))
        all_zero_after_filter.append(
            int(np.sum(np.sum(Y_data_filtered > 0, axis=0) == 0))
            + int(np.sum(np.sum(X_data_filtered > 0, axis=0) == 0)))

        u_tpr, u_f1, u_mse = _arm_c_metrics(p_u, true_lfcs_filtered, lfc_u,
                                            arm.correction)
        l_tpr, l_f1, l_mse = _arm_c_metrics(p_l, true_lfcs_filtered, lfc_l,
                                            arm.correction)
        tpr_u.append(u_tpr)
        f1_u.append(u_f1)
        mse_u.append(u_mse)
        tpr_l.append(l_tpr)
        f1_l.append(l_f1)
        mse_l.append(l_mse)

    if tpr_u:
        headline.append(_headline("tpr_mean", np.mean(tpr_u), np.mean(tpr_l),
                                  arm.headline_decimals))
        headline.append(_headline("f1_mean", np.mean(f1_u), np.mean(f1_l),
                                  arm.headline_decimals))
        # the script prints both MSE figures to four places
        headline.append(_headline("lfc_mse_mean", np.mean(mse_u), np.mean(mse_l), 4))
        headline.append(_headline("lfc_mse_std", np.std(mse_u), np.std(mse_l), 4))

    provenance = {
        "script": arm.script, "seed": 1, "replicates_total": replicates,
        "replicates_evaluated": len(units),
        "input": path.relative_to(REPO_ROOT).as_posix(),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_obs": int(n_obs), "n_genes_after_mt_filter": int(n_genes),
        "n_mt_genes_removed": int(mt_gene_mask.sum()),
        "min_cells_per_group": min_cells_per_group,
        "n_signal_genes": n_signal_genes,
        "utils_entry": "get_DELN_lfcs",
        "all_zero_genes_after_filter": int(sum(all_zero_after_filter)),
        "deln_vs_ln_within_utils": {
            "max_abs_dlfc": max(within_dlfc) if within_dlfc else None,
            "max_abs_dp": max(within_dp) if within_dp else None,
            "detail": "utils.get_DELN_lfcs against utils.get_LN_lfcs on the same "
                      "inputs, trigamma held fixed. Measured 2026-09-09: "
                      "max|dLFC| 1.03e-06, which is NOT the eps term -- with "
                      "n_plus >= 3 guaranteed by the filter, eps ** (1+n_plus) "
                      "is at most 1e-36, and 0 all-zero genes survive, so the "
                      "guard is unreachable. It is the float32 intermediates "
                      "utils.get_LN_lfcs allocates and get_DELN_lfcs does not. "
                      "So arm C was already on the float64 path, and its "
                      "comparison against lntest isolates the trigamma alone.",
        },
    }
    return _finish(units, dlfc_pool, headline, provenance, len(units) == replicates)


# --------------------------------------------------------------------------

def run_arm(arm: Arm, lntest, reps=None):
    """Reproduce one arm's inputs, run both estimators, recompute its headline."""
    adapters = {"A": _arm_a, "B": _arm_b, "C": _arm_c}
    try:
        adapter = adapters[arm.letter]
    except KeyError:
        raise NotImplementedError(
            f"arm {arm.letter} is registered {arm.status} but has no adapter"
        ) from None
    return adapter(arm, lntest, reps=reps)


# --------------------------------------------------------------------------

def _fmt(value):
    return "n/a" if value is None else f"{value:.3e}"


def _print_arm(arm, res):
    agg = res["aggregate"]
    print(f"[arm {arm.letter}] {arm.name}")
    print(f"           {agg['n_units']} units, {agg['n_gene_tests']} gene tests, "
          f"correction {arm.correction}, entry {res['provenance']['utils_entry']}")
    print(f"           max|dLFC| {_fmt(agg['max_abs_dlfc'])}   "
          f"median|dLFC| {_fmt(agg['median_abs_dlfc'])}   "
          f"max|dlog10 p| {_fmt(agg['max_abs_dlog10p'])} "
          f"({agg['n_excluded_from_dlog10p']} genes excluded, p == 0 either way)")
    print(f"           significance disagreements at alpha={ALPHA}: "
          f"{agg['disagree_total']} "
          f"({agg['disagree_utils_only']} utils only, "
          f"{agg['disagree_lntest_only']} lntest only) -- expected nonzero")
    print(f"           NaN utils/lntest: lfc {agg['nan_lfc_utils']}/"
          f"{agg['nan_lfc_lntest']}, p {agg['nan_p_utils']}/{agg['nan_p_lntest']}")
    moved = [h for h in res["headline"] if h["moved"]]
    print(f"           headline: {len(moved)} of {len(res['headline'])} moved at "
          "the manuscript's printed precision")
    for h in moved[:10]:
        print(f"             MOVED {h['metric']}: {h['utils_rounded']} -> "
              f"{h['lntest_rounded']} (raw delta {h['delta']:+.3e})")
    if len(moved) > 10:
        print(f"             ... and {len(moved) - 10} more; see the JSON report")
    if not res["complete"]:
        print("           REDUCED RUN: a smoke test. These headline numbers are "
              "not comparable to the manuscript.")


def _verdict(gate_failures, moved, unmeasured, api_failures=()):
    if api_failures:
        print(f"\nAPI ASSERTION FAILED: {'; '.join(api_failures)}. "
              "These pin the surface stage 3b has to rewire. A failure here "
              "means the refactor's import plan is wrong, so the arm numbers "
              "below are measured against an API that is not the one being "
              "adopted.")
        return 5
    if gate_failures:
        print(f"\nHARD GATE FAILED on arms {', '.join(gate_failures)}: "
              f"max|dLFC| >= {HARD_DLFC_TOL:g}. The log-fold change involves no "
              "trigamma, so something other than that term changed. This is a "
              "failure of the run, not a finding about the estimator.")
        return 3
    if moved:
        print(f"\nHEADLINE NUMBERS MOVED on arms {', '.join(moved)}. "
              "Stop here: that is a manuscript change, not a cleanup, and it "
              "goes to Oskar before anything is deleted. "
              "Do not proceed to stage 3.")
        return 4
    if unmeasured:
        print(f"\nINCOMPLETE: arms {', '.join(unmeasured)} were not measured in "
              "full. No gate has been evaluated for them; this is not a pass.")
        return 2
    print("\nBoth gates pass: max|dLFC| is below tolerance on every measured arm "
          "and no headline number moved at its printed precision. The "
          "disagreement counts above are expected, and they are the finding.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="all",
                    help="arm letter (A-G) or 'all' (default)")
    ap.add_argument("--reps", type=int, default=None,
                    help="override replicate count; the headline metric is only "
                         "comparable to the manuscript at the full count, so a "
                         "reduced run is a smoke test and is labelled as one")
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="write the full report here")
    args = ap.parse_args(argv)

    report = {"frozen_copy": None, "frozen_metric": None,
              "api_mismatches": None, "baseline_rng_neutrality": None,
              "arms": {}}

    state, detail = verify_frozen_copy()
    report["frozen_copy"] = {"state": state, "detail": detail}
    print(f"[frozen copy]   {state}: {detail}")

    state, detail = verify_frozen_metric()
    report["frozen_metric"] = {"state": state, "detail": detail}
    print(f"[frozen metric] {state}: {detail}\n")

    lntest = require_lntest()
    report["api_mismatches"] = check_api_mismatches(lntest)
    for f in report["api_mismatches"]:
        print(f"[api] {'ok  ' if f['ok'] else 'FAIL'} {f['mismatch']}")
        print(f"      {f['detail']}")

    neutrality = check_baseline_rng_neutrality()
    report["baseline_rng_neutrality"] = neutrality
    # ok is True, or None when the check could not be run; a False would have
    # raised inside check_baseline_rng_neutrality rather than reached here
    print(f"[rng] {'ok  ' if neutrality['ok'] else '??  '} {neutrality['check']}")
    print(f"      {neutrality['detail']}\n")

    selected = [a for a in ARMS if args.arm == "all" or a.letter == args.arm.upper()]
    if not selected:
        raise SystemExit(f"no such arm: {args.arm}")

    unmeasured, gate_failures, moved = [], [], []
    for arm in selected:
        entry = asdict(arm)
        entry["result"] = None
        if arm.status != MEASURABLE:
            # NOT_APPLICABLE is a proof of no-effect -- the arm never imports
            # utils, so there is nothing to measure and nothing to report.
            # COVERED_BY_ARGUMENT is not: it is an argument standing in for a
            # measurement that has not happened, so it counts as unmeasured and
            # the run must not exit 0. Collapsing the two is how arms D and E
            # came to sit behind a "Both gates pass" line.
            print(f"[arm {arm.letter}] {arm.status}: {arm.name}")
            print(f"           {arm.note}")
            if arm.status == COVERED_BY_ARGUMENT:
                unmeasured.append(arm.letter)
        else:
            try:
                entry["result"] = run_arm(arm, lntest, reps=args.reps)
            except (NotImplementedError, MissingInput) as exc:
                print(f"[arm {arm.letter}] UNMEASURED: {exc}")
                unmeasured.append(arm.letter)
            else:
                res = entry["result"]
                _print_arm(arm, res)
                if not res["hard_gate"]["ok"]:
                    gate_failures.append(arm.letter)
                if not res["complete"]:
                    unmeasured.append(arm.letter)
                elif res["headline_moved"]:
                    # a reduced run's headline is not the manuscript's, so it
                    # cannot be said to have moved -- it is unmeasured instead
                    moved.append(arm.letter)
        report["arms"][arm.letter] = entry
        print()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json}")

    api_failures = [f["mismatch"] for f in report["api_mismatches"] if not f["ok"]]
    return _verdict(gate_failures, moved, unmeasured, api_failures)

if __name__ == "__main__":
    sys.exit(main())
