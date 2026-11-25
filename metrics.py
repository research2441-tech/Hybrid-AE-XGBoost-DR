"""
metrics.py
----------
Evaluation measures, confidence intervals, clinical error analysis, figures and
statistical significance tests (Reviewer 1, comments 7, 15 and 18).

Reported for every evaluation
  * overall: accuracy, balanced accuracy, macro precision / recall / F1,
    weighted F1, Cohen's kappa (unweighted), quadratic weighted kappa (QWK),
    Matthews correlation coefficient, macro one-vs-rest ROC-AUC;
  * class-wise (one-vs-rest): precision, sensitivity (recall), specificity,
    negative predictive value, F1-score and support;
  * clinical: referable-DR (grade >= 2) sensitivity / specificity,
    under-grading rate, adjacent-grade error share, and the proportion of
    Severe / PDR images assigned to No DR or Mild (critical false negatives);
  * percentile bootstrap 95 % confidence intervals.

Statistical tests
  * McNemar's test (paired predictions on one test set);
  * Friedman test across classifiers over repeated CV folds;
  * Wilcoxon signed-rank test with Holm-Bonferroni correction;
  * Nadeau-Bengio corrected resampled t-test for repeated k-fold CV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from common import CLASS_NAMES, NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------
def overall_metrics(y: np.ndarray, p: np.ndarray, proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    m = {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_precision": precision_score(y, p, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(y, p, average="macro", labels=LABELS, zero_division=0),
        "macro_f1": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", labels=LABELS, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y, p, labels=LABELS),
        "quadratic_weighted_kappa": cohen_kappa_score(y, p, labels=LABELS, weights="quadratic"),
        "mcc": matthews_corrcoef(y, p),
    }
    cm = confusion_matrix(y, p, labels=LABELS)
    spec = []
    for c in LABELS:
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
    m["macro_specificity"] = float(np.nanmean(spec))
    if proba is not None and len(np.unique(y)) > 1:
        try:
            present = np.unique(y)
            m["macro_auc_ovr"] = roc_auc_score(np.eye(NUM_CLASSES)[y][:, present], proba[:, present], average="macro")
        except ValueError:
            m["macro_auc_ovr"] = np.nan
    return {k: float(v) for k, v in m.items()}


def classwise_metrics(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y, p, labels=LABELS)
    rows = []
    for c in LABELS:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        sens = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"class": CLASS_NAMES[c], "support": int(tp + fn), "TP": int(tp), "FP": int(fp),
                     "FN": int(fn), "TN": int(tn), "precision": prec, "sensitivity": sens,
                     "specificity": tn / (tn + fp) if tn + fp else 0.0,
                     "npv": tn / (tn + fn) if tn + fn else 0.0,
                     "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0})
    return pd.DataFrame(rows)


def clinical_metrics(y: np.ndarray, p: np.ndarray, referable: Sequence[int] = (2, 3, 4)) -> Dict[str, float]:
    """Screening-oriented error analysis for the five-grade scale."""
    ref = np.isin(y, referable)
    ref_hat = np.isin(p, referable)
    tp = int((ref & ref_hat).sum()); fn = int((ref & ~ref_hat).sum())
    tn = int((~ref & ~ref_hat).sum()); fp = int((~ref & ref_hat).sum())
    wrong = y != p
    adv = np.isin(y, [3, 4])
    any_dr = y > 0
    return {
        "referable_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "referable_specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "referable_false_negatives": fn,
        "undergrading_rate": float((p < y).mean()),
        "overgrading_rate": float((p > y).mean()),
        "adjacent_grade_error_share": float((np.abs(p - y)[wrong] == 1).mean()) if wrong.any() else 0.0,
        "any_dr_predicted_as_no_dr": float((p[any_dr] == 0).mean()) if any_dr.any() else 0.0,
        "severe_or_pdr_predicted_as_no_dr_or_mild": float((p[adv] <= 1).mean()) if adv.any() else 0.0,
        "severe_or_pdr_count": int(adv.sum()),
    }


def evaluate(y: np.ndarray, p: np.ndarray, proba: Optional[np.ndarray] = None,
             referable: Sequence[int] = (2, 3, 4)) -> Dict:
    return {"overall": overall_metrics(y, p, proba), "classwise": classwise_metrics(y, p).to_dict("records"),
            "clinical": clinical_metrics(y, p, referable),
            "confusion_matrix": confusion_matrix(y, p, labels=LABELS).tolist(), "n_test": int(len(y))}


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------
def bootstrap_ci(y: np.ndarray, p: np.ndarray, proba: Optional[np.ndarray], n_boot: int = 2000,
                 level: float = 0.95, seed: int = 42) -> Dict[str, List[float]]:
    """Stratified percentile bootstrap confidence intervals of the overall metrics."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in LABELS]
    samples: Dict[str, List[float]] = {}
    for _ in range(int(n_boot)):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class if len(ix)])
        m = overall_metrics(y[idx], p[idx], None if proba is None else proba[idx])
        for k, v in m.items():
            samples.setdefault(k, []).append(v)
    a = (1 - level) / 2
    return {k: [float(np.nanquantile(v, a)), float(np.nanquantile(v, 1 - a))] for k, v in samples.items()}


def mean_sd_ci(values: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    """Mean, sample SD and Student-t confidence interval over repeated runs."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    half = stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half, "n": n}


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def mcnemar(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> Dict[str, float]:
    a_ok, b_ok = p_a == y, p_b == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "none (identical errors)"}
    if b + c < 25:
        pv = float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": pv, "test": "exact binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "test": "chi-square with continuity correction"}


def corrected_resampled_ttest(a: Sequence[float], b: Sequence[float], n_train: int, n_test: int) -> Dict[str, float]:
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test for paired scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    if k < 2 or np.allclose(d.var(ddof=1), 0):
        return {"t": 0.0, "p_value": 1.0, "mean_diff": float(d.mean()) if k else 0.0}
    var = d.var(ddof=1) * (1.0 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return {"t": float(t), "p_value": float(2 * stats.t.sf(abs(t), k - 1)), "mean_diff": float(d.mean())}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def friedman_wilcoxon(scores: pd.DataFrame, reference: str) -> Dict:
    """Friedman omnibus test + Holm-corrected Wilcoxon tests vs the reference column."""
    out: Dict = {}
    cols = list(scores.columns)
    if len(cols) >= 3:
        stat, p = stats.friedmanchisquare(*[scores[c].to_numpy() for c in cols])
        out["friedman"] = {"statistic": float(stat), "p_value": float(p)}
    raw = {}
    for c in cols:
        if c == reference:
            continue
        diff = scores[reference] - scores[c]
        raw[c] = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(scores[reference], scores[c],
                                                                       zero_method="zsplit").pvalue)
    out["wilcoxon_raw"] = raw
    out["wilcoxon_holm"] = holm(raw) if raw else {}
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    plt = _plt()
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, ttl, fmt in ((axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised (recall)", ".2f")):
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = format(mat[i, j], fmt)
                ax.text(j, i, val, ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else "#1a1a1a")
        ax.set_xticks(LABELS); ax.set_yticks(LABELS)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted grade"); ax.set_ylabel("Reference grade"); ax.set_title(ttl)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(title)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y: np.ndarray, proba: np.ndarray, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 5))
    colors = ["#2a6f97", "#61a5c2", "#e9c46a", "#f4a261", "#c1121f"]
    for c in LABELS:
        yt = (y == c).astype(int)
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, proba[:, c])
        auc = roc_auc_score(yt, proba[:, c])
        ax.plot(fpr, tpr, color=colors[c], lw=1.8, label=f"{CLASS_NAMES[c]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)"); ax.set_ylabel("Sensitivity")
    ax.set_title(title); ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_evaluation(res: Dict, out_dir: Path, prefix: str, title: str,
                    y: Optional[np.ndarray] = None, proba: Optional[np.ndarray] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["classwise"]).to_csv(out_dir / f"{prefix}_classwise.csv", index=False)
    pd.DataFrame(res["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES) \
        .to_csv(out_dir / f"{prefix}_confusion_matrix.csv")
    plot_confusion(np.asarray(res["confusion_matrix"]), out_dir / f"{prefix}_confusion_matrix.png", title)
    if y is not None and proba is not None:
        plot_roc(y, proba, out_dir / f"{prefix}_roc.png", title)
