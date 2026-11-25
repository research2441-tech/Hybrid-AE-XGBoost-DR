"""
feature_pipeline.py
-------------------
Hybrid feature construction, training-only feature selection / PCA and the
XGBoost classification stage of the proposed framework.

Proposed model (the only configuration referred to as "proposed"):

  fundus image -> preprocessing -> [ AE latent z (256) || handcrafted h (73) ]
               -> median imputation -> Pearson redundancy filter (|r| <= tau)
               -> z-score standardisation -> PCA (95 % variance)
               -> XGBoost (multi:softprob, 5 classes) -> DR grade

Every data-dependent step (imputation statistics, Pearson coefficients,
standardisation, PCA eigenvectors, XGBoost trees) is fitted on the TRAINING
partition only, inside a scikit-learn Pipeline, and then applied unchanged to
the validation / test partitions (Reviewer 1, comment 13). The Pearson
threshold tau is chosen from a predefined grid by validation macro-F1; the test
partition is never consulted.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import LOGGER, NUM_CLASSES, cache_dir
from handcrafted_features import extract_batch, feature_names
from preprocessing import augment_image, config_fingerprint, load_cached


# ==========================================================================
# 1. Pearson correlation feature selector (fitted on training data only)
# ==========================================================================
class PearsonCorrelationSelector(BaseEstimator, TransformerMixin):
    """Relevance-ordered redundancy filter based on Pearson's r.

    fit():
      1. drop zero-variance features;
      2. relevance_j = |r(x_j, y)| with y the ordinal DR grade (0-4);
      3. optionally drop features with relevance below ``min_relevance``;
      4. visit features in decreasing relevance and keep feature j only if
         |r(x_j, x_k)| <= threshold for every already-kept feature k.
    All coefficients are computed from the matrix passed to fit(), i.e. the
    training partition.
    """

    def __init__(self, threshold: float = 0.95, min_relevance: float = 0.0):
        self.threshold = threshold
        self.min_relevance = min_relevance

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        std = X.std(axis=0)
        valid = std > 1e-12
        Xc = (X[:, valid] - X[:, valid].mean(axis=0)) / std[valid]
        yc = (y - y.mean()) / (y.std() + 1e-12)
        relevance = np.abs(Xc.T @ yc / len(y))
        corr = np.abs(np.corrcoef(Xc, rowvar=False)) if Xc.shape[1] > 1 else np.ones((1, 1))
        corr = np.nan_to_num(corr)
        idx_valid = np.flatnonzero(valid)
        order = np.argsort(-relevance, kind="stable")
        kept: List[int] = []
        for j in order:
            if relevance[j] < self.min_relevance:
                continue
            if all(corr[j, k] <= self.threshold for k in kept):
                kept.append(j)
        if not kept:                       # always keep at least the most relevant feature
            kept = [int(order[0])]
        support = np.zeros(X.shape[1], dtype=bool)
        support[idx_valid[np.array(sorted(kept))]] = True
        self.support_ = support
        self.relevance_ = np.zeros(X.shape[1])
        self.relevance_[idx_valid] = relevance
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


# ==========================================================================
# 2. Classifiers
# ==========================================================================
XGB_KEYS = ("objective", "num_class", "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "gamma", "reg_lambda", "reg_alpha", "tree_method", "eval_metric")


def make_classifier(name: str, cfg: Dict, seed: int, xgb_overrides: Optional[Dict] = None,
                    early_stopping: bool = True):
    """Factory for the proposed classifier (xgboost) and the comparison classifiers."""
    nj = cfg.get("n_jobs", -1)
    if name == "xgboost":
        params = {k: cfg["xgboost"][k] for k in XGB_KEYS}
        params.update(xgb_overrides or {})
        es = int(cfg["xgboost"]["early_stopping_rounds"]) if early_stopping else None
        return XGBClassifier(**params, random_state=seed, n_jobs=nj, early_stopping_rounds=es, verbosity=0)
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=nj)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=nj)
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


CLASSIFIER_LABELS = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
                     "knn": "K-nearest neighbours", "naive_bayes": "Gaussian naive Bayes",
                     "random_forest": "Random forest", "svm": "SVM (RBF)", "xgboost": "XGBoost"}


# ==========================================================================
# 3. Pipeline construction and training-only fitting
# ==========================================================================
def build_preprocessor(cfg: Dict, use_selection: bool = True, use_pca: bool = True,
                       threshold: Optional[float] = None) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if use_selection:
        fs = cfg["feature_selection"]
        steps.append(("pearson", PearsonCorrelationSelector(
            threshold=float(threshold if threshold is not None else fs["redundancy_threshold"]),
            min_relevance=float(fs.get("min_relevance", 0.0)))))
    steps.append(("scale", StandardScaler()))
    if use_pca and cfg["pca"].get("enabled", True):
        steps.append(("pca", PCA(n_components=float(cfg["pca"]["variance_retained"]),
                                 whiten=bool(cfg["pca"].get("whiten", False)),
                                 svd_solver="full", random_state=int(cfg["seed"]))))
    return Pipeline(steps)


def fit_model(Xtr: np.ndarray, ytr: np.ndarray, Xva: Optional[np.ndarray], yva: Optional[np.ndarray],
              cfg: Dict, classifier: str = "xgboost", use_selection: bool = True, use_pca: bool = True,
              threshold: Optional[float] = None, xgb_overrides: Optional[Dict] = None,
              seed: Optional[int] = None) -> Tuple[Pipeline, Dict]:
    """Fit preprocessing + classifier on TRAINING data; validation only for early stopping."""
    seed = int(cfg["seed"] if seed is None else seed)
    pre = build_preprocessor(cfg, use_selection, use_pca, threshold)
    info: Dict = {}
    t0 = time.perf_counter()
    Xtr_t = pre.fit_transform(Xtr, ytr)
    info["feature_transform_fit_s"] = time.perf_counter() - t0
    info["n_input_features"] = int(Xtr.shape[1])
    info["n_selected_features"] = int(pre.named_steps["pearson"].support_.sum()) if use_selection else int(Xtr.shape[1])
    info["n_model_features"] = int(Xtr_t.shape[1])
    if "pca" in pre.named_steps:
        info["pca_components"] = int(pre.named_steps["pca"].n_components_)
        info["pca_explained_variance"] = float(pre.named_steps["pca"].explained_variance_ratio_.sum())

    use_es = classifier == "xgboost" and Xva is not None and len(Xva) > 0
    clf = make_classifier(classifier, cfg, seed, xgb_overrides, early_stopping=use_es)
    t0 = time.perf_counter()
    if classifier == "xgboost":
        w = compute_sample_weight("balanced", ytr) if cfg["xgboost"].get("class_weighting") == "balanced" else None
        if use_es:
            clf.fit(Xtr_t, ytr, sample_weight=w, eval_set=[(pre.transform(Xva), yva)], verbose=False)
            info["best_iteration"] = int(clf.best_iteration)
        else:
            clf.fit(Xtr_t, ytr, sample_weight=w, verbose=False)
    else:
        clf.fit(Xtr_t, ytr)
    info["classifier_fit_s"] = time.perf_counter() - t0
    info["train_time_s"] = info["feature_transform_fit_s"] + info["classifier_fit_s"]
    model = Pipeline(pre.steps + [("clf", clf)])
    return model, info


def predict(model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, probabilities, seconds)."""
    t0 = time.perf_counter()
    proba = model.predict_proba(X)
    dt = time.perf_counter() - t0
    full = np.zeros((len(X), NUM_CLASSES))
    classes = model.named_steps["clf"].classes_
    full[:, classes] = proba
    return full.argmax(axis=1), full, dt


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))


def tune_on_validation(Xtr, ytr, Xva, yva, cfg: Dict, out_dir: Optional[Path] = None) -> Dict:
    """Select the Pearson threshold, then XGBoost hyperparameters, by validation macro-F1."""
    fs, xg = cfg["feature_selection"], cfg["xgboost"]
    best = {"threshold": float(fs["redundancy_threshold"]), "xgb": {}}
    rows = []
    if fs.get("tune_threshold", True) and len(Xva):
        scores = {}
        for tau in fs["threshold_grid"]:
            m, info = fit_model(Xtr, ytr, Xva, yva, cfg, threshold=tau)
            s = macro_f1(yva, predict(m, Xva)[0])
            scores[tau] = s
            rows.append({"stage": "pearson_threshold", "threshold": tau, "val_macro_f1": s,
                         "n_selected": info["n_selected_features"], "pca_components": info.get("pca_components")})
            LOGGER.info("  tau=%.2f  selected=%d  PCA=%s  val macro-F1=%.4f", tau, info["n_selected_features"],
                        info.get("pca_components"), s)
        best["threshold"] = float(max(scores, key=lambda t: (scores[t], -t)))
    if xg.get("tune", True) and len(Xva):
        grid = xg["tuning_grid"]
        keys = list(grid)
        scores = {}
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            m, _ = fit_model(Xtr, ytr, Xva, yva, cfg, threshold=best["threshold"], xgb_overrides=params)
            s = macro_f1(yva, predict(m, Xva)[0])
            scores[combo] = s
            rows.append({"stage": "xgboost_grid", "threshold": best["threshold"], **params, "val_macro_f1": s})
        best_combo = max(scores, key=scores.get)
        best["xgb"] = dict(zip(keys, best_combo))
    best["validation_table"] = rows
    if out_dir is not None and rows:
        pd.DataFrame(rows).to_csv(out_dir / "validation_tuning.csv", index=False)
    LOGGER.info("Selected Pearson threshold=%.2f, XGBoost overrides=%s", best["threshold"], best["xgb"])
    return best


# ==========================================================================
# 4. Feature bank: handcrafted + AE features for originals and augmented copies
# ==========================================================================
def _aug_key(parent: str, k: int) -> str:
    return f"{parent}|{k}"


def _load_original(manifest: pd.DataFrame, idx: Sequence[int]) -> np.ndarray:
    return np.stack([load_cached(manifest["cache_path"].iloc[i]) for i in idx])


def _load_augmented(path_of: Dict[str, str], items: Sequence[Tuple[str, int, int]], cfg: Dict) -> np.ndarray:
    seed = int(cfg["seed"])
    return np.stack([augment_image(load_cached(path_of[p]), p, k, cfg["augmentation"], seed)[0]
                     for p, k, _ in items])


def handcrafted_bank(cfg: Dict, key: str, manifest: pd.DataFrame,
                     plan: Sequence[Tuple[str, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Handcrafted features of originals and of the requested augmented copies (disk-cached)."""
    hcfg = cfg["handcrafted"]
    fp = config_fingerprint({**cfg["preprocessing"], **{f"hc_{k}": str(v) for k, v in hcfg.items()},
                             **{f"aug_{k}": str(v) for k, v in cfg["augmentation"].items()}})
    folder = cache_dir(cfg, key)
    f_orig = folder / f"handcrafted_originals_{fp}.npz"
    if f_orig.exists():
        d = np.load(f_orig, allow_pickle=True)
        cached = dict(zip(d["ids"].tolist(), d["X"]))
    else:
        cached = {}
    missing = [i for i, iid in enumerate(manifest["image_id"]) if iid not in cached]
    if missing:
        LOGGER.info("[%s] handcrafted features for %d original images", key, len(missing))
        for s in range(0, len(missing), 1024):
            chunk = missing[s:s + 1024]
            X = extract_batch(list(_load_original(manifest, chunk)), hcfg, cfg.get("n_jobs", -1))
            for i, row in zip(chunk, X):
                cached[manifest["image_id"].iloc[i]] = row
        np.savez_compressed(f_orig, ids=np.array(list(cached), dtype=object), X=np.vstack(list(cached.values())))
    X_orig = np.vstack([cached[i] for i in manifest["image_id"]])

    f_aug = folder / f"handcrafted_augmented_{fp}.npz"
    if f_aug.exists():
        d = np.load(f_aug, allow_pickle=True)
        acached = dict(zip(d["ids"].tolist(), d["X"]))
    else:
        acached = {}
    todo = [it for it in plan if _aug_key(it[0], it[1]) not in acached]
    if todo:
        LOGGER.info("[%s] handcrafted features for %d augmented training images", key, len(todo))
        path_of = dict(zip(manifest["image_id"], manifest["cache_path"]))
        for s in range(0, len(todo), 1024):
            chunk = todo[s:s + 1024]
            X = extract_batch(list(_load_augmented(path_of, chunk, cfg)), hcfg, cfg.get("n_jobs", -1))
            for it, row in zip(chunk, X):
                acached[_aug_key(it[0], it[1])] = row
        np.savez_compressed(f_aug, ids=np.array(list(acached), dtype=object), X=np.vstack(list(acached.values())))
    nf = X_orig.shape[1]
    X_aug = np.vstack([acached[_aug_key(p, k)] for p, k, _ in plan]) if plan else np.zeros((0, nf), np.float32)
    return X_orig, X_aug


def autoencoder_bank(encoder, cfg: Dict, manifest: pd.DataFrame,
                     plan: Sequence[Tuple[str, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Latent AE features of originals and augmented copies for a given trained encoder."""
    from autoencoder import encode
    size = int(cfg["autoencoder"]["input_size"])
    orig = []
    for s in range(0, len(manifest), 512):
        idx = list(range(s, min(s + 512, len(manifest))))
        orig.append(encode(encoder, _load_original(manifest, idx), size))
    X_orig = np.vstack(orig)
    aug = []
    path_of = dict(zip(manifest["image_id"], manifest["cache_path"]))
    for s in range(0, len(plan), 512):
        aug.append(encode(encoder, _load_augmented(path_of, plan[s:s + 512], cfg), size))
    X_aug = np.vstack(aug) if aug else np.zeros((0, X_orig.shape[1]), np.float32)
    return X_orig, X_aug


def hybrid_names(cfg: Dict) -> List[str]:
    return [f"ae_{i}" for i in range(int(cfg["autoencoder"]["latent_dim"]))] + feature_names(cfg["handcrafted"])


def compose(ae: np.ndarray, hc: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ae":
        return ae
    if mode == "hc":
        return hc
    if mode == "hybrid":
        return np.hstack([ae, hc])
    raise ValueError(mode)


def training_matrix(ae_o, hc_o, ae_a, hc_a, manifest: pd.DataFrame, plan, train_mask: np.ndarray,
                    mode: str, use_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Training features = training originals (+ their augmented copies)."""
    y = manifest["label"].to_numpy()
    X = compose(ae_o[train_mask], hc_o[train_mask], mode)
    yy = y[train_mask]
    if use_augmentation and len(plan):
        train_ids = set(manifest["image_id"][train_mask])
        keep = np.array([p in train_ids for p, _, _ in plan])
        if (~keep).any():
            raise RuntimeError("Augmented copy of a non-training image requested: leakage guard triggered.")
        X = np.vstack([X, compose(ae_a, hc_a, mode)])
        yy = np.concatenate([yy, np.array([lab for _, _, lab in plan])])
    return X, yy
