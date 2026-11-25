"""
ablation.py
-----------
Component-wise ablation of the proposed framework (Reviewer 1, comment 8).

All variants use the identical train / validation / test partition, the same
preprocessing, the same AE weights and the hyperparameters selected for the
full model on the validation set, so the only difference between rows is the
removed component.

  A1  AE latent features only            (+ Pearson + PCA + XGBoost)
  A2  Handcrafted features only          (+ Pearson + PCA + XGBoost)
  A3  Hybrid features, no Pearson selection
  A4  Hybrid features, no PCA
  A5  Hybrid features, no Pearson and no PCA
  A6  Hybrid features, no training augmentation
  A7  Full proposed framework

For each variant: accuracy, balanced accuracy, macro-F1, Cohen's kappa, QWK
with bootstrap 95 % CIs, feature dimensionality, training time, and McNemar's
test against the full model.

Usage
  python ablation.py --dataset aptos
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from common import LOGGER, base_argparser, dataset_keys, load_config, load_json, output_dir, set_global_seed, \
    setup_logging, save_json
from feature_pipeline import compose, fit_model, predict, training_matrix
from metrics import _plt, bootstrap_ci, mcnemar, overall_metrics
from train_hybrid import prepare_features

VARIANTS = [
    ("A1", "AE features only", "ae", True, True, True),
    ("A2", "Handcrafted features only", "hc", True, True, True),
    ("A3", "Hybrid, without Pearson selection", "hybrid", False, True, True),
    ("A4", "Hybrid, without PCA", "hybrid", True, False, True),
    ("A5", "Hybrid, without Pearson and PCA", "hybrid", False, False, True),
    ("A6", "Hybrid, without training augmentation", "hybrid", True, True, False),
    ("A7", "Full proposed framework", "hybrid", True, True, True),
]


def run(cfg: Dict, key: str) -> pd.DataFrame:
    LOGGER.info("=== Ablation study | dataset: %s ===", key)
    out = output_dir(cfg, key, "ablation")
    bp_file = Path(cfg["paths"]["output_root"]) / key / "proposed" / "best_params.json"
    bp = load_json(bp_file) if bp_file.exists() else {"pearson_threshold": cfg["feature_selection"]["redundancy_threshold"]}
    tau = bp.pop("pearson_threshold")
    F = prepare_features(cfg, key)
    man, plan, m = F["manifest"], F["plan"], F["masks"]
    y = man["label"].to_numpy()
    yva, yte = y[m["val"]], y[m["test"]]

    rows, preds = [], {}
    for code, name, mode, use_fs, use_pca, use_aug in VARIANTS:
        Xtr, ytr = training_matrix(F["ae_o"], F["hc_o"], F["ae_a"], F["hc_a"], man, plan, m["train"], mode, use_aug)
        Xo = compose(F["ae_o"], F["hc_o"], mode)
        model, info = fit_model(Xtr, ytr, Xo[m["val"]], yva, cfg, "xgboost", use_fs, use_pca, tau, bp)
        pred, proba, t_inf = predict(model, Xo[m["test"]])
        preds[code] = pred
        ov = overall_metrics(yte, pred, proba)
        ci = bootstrap_ci(yte, pred, proba, min(1000, cfg["evaluation"]["bootstrap_iterations"]),
                          cfg["evaluation"]["confidence_level"], int(cfg["seed"]))
        rows.append({"code": code, "variant": name, "features_in": info["n_input_features"],
                     "features_after_pearson": info["n_selected_features"],
                     "features_to_classifier": info["n_model_features"], "train_samples": len(ytr),
                     **{k: ov[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                                            "quadratic_weighted_kappa")},
                     "macro_f1_ci_low": ci["macro_f1"][0], "macro_f1_ci_high": ci["macro_f1"][1],
                     "accuracy_ci_low": ci["accuracy"][0], "accuracy_ci_high": ci["accuracy"][1],
                     "train_time_s": info["train_time_s"],
                     "inference_ms_per_image": 1000 * t_inf / max(len(yte), 1)})
        LOGGER.info("  %s %-42s acc=%.4f  macro-F1=%.4f  dims=%d", code, name, ov["accuracy"], ov["macro_f1"],
                    info["n_model_features"])

    tab = pd.DataFrame(rows)
    tests = {c: mcnemar(yte, preds["A7"], preds[c]) for c in preds if c != "A7"}
    tab["mcnemar_p_vs_full"] = tab["code"].map(lambda c: tests[c]["p_value"] if c in tests else np.nan)
    tab.to_csv(out / "ablation_results.csv", index=False)
    save_json({"pearson_threshold": tau, "xgb_params": bp, "mcnemar_vs_full": tests}, out / "ablation_tests.json")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    err = np.vstack([tab["macro_f1"] - tab["macro_f1_ci_low"], tab["macro_f1_ci_high"] - tab["macro_f1"]])
    colors = ["#9db4c0"] * (len(tab) - 1) + ["#1d3557"]
    ax.barh(tab["code"] + "  " + tab["variant"], tab["macro_f1"], xerr=err, color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 on the held-out test set (95 % bootstrap CI)")
    ax.set_title(f"Ablation study — {key.upper()}")
    fig.savefig(out / "ablation_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tab


def main() -> None:
    args = base_argparser("Component ablation of the proposed framework.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "ablation.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        run(cfg, key)


if __name__ == "__main__":
    main()
