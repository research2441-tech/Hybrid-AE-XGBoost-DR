"""
repeated_cv.py
--------------
Repeated stratified group k-fold cross-validation of the complete framework
(Reviewer 1, comments 6 and 15).

For every repeat r and fold k (default 5 folds x 3 repeats = 15 runs):
  * outer split: StratifiedGroupKFold(seed + r) over ORIGINAL images, using
    the same groups as data_preparation.py (patient / near-duplicate groups);
  * inner validation: 10 % of the training fold (group-aware) for AE early
    stopping and XGBoost early stopping;
  * augmentation plan computed from the training fold only;
  * autoencoder retrained on the training fold (configurable);
  * Pearson filter, scaler, PCA and classifier fitted on the training fold only;
  * hyperparameters fixed a priori to the values in config.yaml, so no
    information from any test fold influences model selection.

Outputs
  cv_fold_results.csv       one row per (repeat, fold, classifier)
  cv_summary.csv            mean ± SD and t-based 95 % CI per metric
  cv_significance.json      Friedman, Holm-corrected Wilcoxon and
                            Nadeau-Bengio corrected t-tests vs XGBoost
  cv_boxplot.png

Usage
  python repeated_cv.py --dataset aptos
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (LOGGER, base_argparser, dataset_keys, environment_info, load_config,
                    output_dir, save_json, set_global_seed, setup_logging)
from data_preparation import load_manifest
from feature_pipeline import (CLASSIFIER_LABELS, autoencoder_bank, compose, fit_model, handcrafted_bank, predict,
                              training_matrix)
from metrics import _plt, corrected_resampled_ttest, friedman_wilcoxon, mean_sd_ci, overall_metrics
from preprocessing import augmentation_plan

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "quadratic_weighted_kappa"]


def run(cfg: Dict, key: str) -> pd.DataFrame:
    from autoencoder import load_models, train_autoencoder

    LOGGER.info("=== Repeated stratified group CV | dataset: %s ===", key)
    cv = cfg["repeated_cv"]
    out = output_dir(cfg, key, "repeated_cv")
    man = load_manifest(cfg, key)
    y = man["label"].to_numpy()
    groups = man["group"].to_numpy()
    seed = int(cfg["seed"])
    rows: List[Dict] = []

    shared_encoder = None
    if not cv.get("retrain_autoencoder_per_fold", True):
        shared_encoder, _ = load_models(Path(cfg["paths"]["output_root"]) / key / "autoencoder")
        LOGGER.warning("Using the AE trained on the main training partition inside CV folds "
                       "(retrain_autoencoder_per_fold=false).")

    for r in range(int(cv["n_repeats"])):
        outer = StratifiedGroupKFold(n_splits=int(cv["n_splits"]), shuffle=True, random_state=seed + r)
        for k, (tr_idx, te_idx) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
            fold_seed = seed + 100 * r + k
            inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=fold_seed)
            fit_rel, va_rel = next(inner.split(tr_idx, y[tr_idx], groups[tr_idx]))
            fit_idx, va_idx = tr_idx[fit_rel], tr_idx[va_rel]
            assert not set(groups[fit_idx]) & set(groups[te_idx]), "group leakage between CV folds"

            fold_man = man.copy()
            fold_man["split"] = "unused"
            fold_man.loc[fit_idx, "split"] = "train"
            fold_man.loc[va_idx, "split"] = "val"
            fold_man.loc[te_idx, "split"] = "test"
            tr_mask = (fold_man["split"] == "train").to_numpy()
            plan = augmentation_plan(y[tr_mask], fold_man["image_id"][tr_mask], cfg["augmentation"], fold_seed)

            t0 = time.perf_counter()
            if shared_encoder is None:
                fdir = output_dir(cfg, key, "repeated_cv", f"ae_r{r}_k{k}")
                encoder, _, _ = train_autoencoder(fold_man["cache_path"][tr_mask].tolist(),
                                                  fold_man["cache_path"][va_idx].tolist(), cfg["autoencoder"],
                                                  fdir, fold_seed, epochs=int(cv.get("autoencoder_epochs", 30)))
            else:
                encoder = shared_encoder
            ae_time = time.perf_counter() - t0
            hc_o, hc_a = handcrafted_bank(cfg, key, man, plan)
            ae_o, ae_a = autoencoder_bank(encoder, cfg, man, plan)
            Xtr, ytr = training_matrix(ae_o, hc_o, ae_a, hc_a, fold_man, plan, tr_mask, "hybrid")
            Xo = compose(ae_o, hc_o, "hybrid")

            for name in cv["classifiers"]:
                model, info = fit_model(Xtr, ytr, Xo[va_idx], y[va_idx], cfg, name, seed=fold_seed)
                pred, proba, t_inf = predict(model, Xo[te_idx])
                ov = overall_metrics(y[te_idx], pred, proba)
                rows.append({"repeat": r, "fold": k, "classifier": name, "n_train_original": int(tr_mask.sum()),
                             "n_train_total": int(len(ytr)), "n_test": int(len(te_idx)),
                             **{mm: ov[mm] for mm in METRICS}, "train_time_s": info["train_time_s"],
                             "ae_train_time_s": ae_time,
                             "inference_ms_per_image": 1000 * t_inf / max(len(te_idx), 1)})
                LOGGER.info("  repeat %d fold %d %-20s acc=%.4f macro-F1=%.4f", r, k, name, ov["accuracy"],
                            ov["macro_f1"])
            pd.DataFrame(rows).to_csv(out / "cv_fold_results.csv", index=False)

    df = pd.DataFrame(rows)
    summ = []
    for name, g in df.groupby("classifier", sort=False):
        row = {"classifier": CLASSIFIER_LABELS.get(name, name), "key": name}
        for mm in METRICS + ["train_time_s", "inference_ms_per_image"]:
            s = mean_sd_ci(g[mm], cfg["evaluation"]["confidence_level"])
            row[f"{mm}_mean"], row[f"{mm}_sd"] = s["mean"], s["sd"]
            row[f"{mm}_ci_low"], row[f"{mm}_ci_high"] = s["ci_low"], s["ci_high"]
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "cv_summary.csv", index=False)

    sig: Dict = {"n_runs": int(df.groupby("classifier").size().iloc[0]), "environment": environment_info()}
    if "xgboost" in df["classifier"].unique() and df["classifier"].nunique() > 1:
        n_tr = int(df["n_train_original"].mean()); n_te = int(df["n_test"].mean())
        for mm in ("macro_f1", "accuracy", "quadratic_weighted_kappa"):
            wide = df.pivot_table(index=["repeat", "fold"], columns="classifier", values=mm)
            res = friedman_wilcoxon(wide, "xgboost")
            res["nadeau_bengio"] = {c: corrected_resampled_ttest(wide["xgboost"], wide[c], n_tr, n_te)
                                    for c in wide.columns if c != "xgboost"}
            sig[mm] = res
    save_json(sig, out / "cv_significance.json")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    order = list(df["classifier"].unique())
    data = [df[df["classifier"] == c]["macro_f1"].to_numpy() for c in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor("#1d3557" if c == "xgboost" else "#c9d6df")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([CLASSIFIER_LABELS.get(c, c) for c in order], rotation=15)
    ax.set_ylabel("Macro-F1 per fold")
    ax.set_title(f"{cv['n_splits']}-fold x {cv['n_repeats']} repeated stratified group CV — {key.upper()}")
    fig.savefig(out / "cv_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("[%s] CV summary:\n%s", key, summary[["classifier", "accuracy_mean", "accuracy_sd",
                                                      "macro_f1_mean", "macro_f1_sd"]].to_string(index=False))
    return summary


def main() -> None:
    args = base_argparser("Repeated stratified group k-fold cross-validation.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "repeated_cv.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        run(cfg, key)


if __name__ == "__main__":
    main()
