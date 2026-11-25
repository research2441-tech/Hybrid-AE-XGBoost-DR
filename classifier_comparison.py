"""
classifier_comparison.py
------------------------
Comparison of the final XGBoost stage with conventional classifiers trained on
the SAME hybrid feature representation (same Pearson threshold, same PCA
setting, same partitions), together with an unambiguous processing-time
analysis (Reviewer 1, comments 11 and 14; Reviewer 2, comment 3).

Only XGBoost belongs to the proposed framework. Logistic regression, decision
tree, K-nearest neighbours, Gaussian naive Bayes, random forest and SVM are
comparison methods.

Timing definitions (single unit per column)
  train_time_s                 fitting of Pearson filter + scaler + PCA + classifier
                               on the full training matrix, in SECONDS
  inference_ms_per_image       median over `evaluation.timing_repeats` runs of the
                               time to predict the whole test set, divided by
                               the number of test images, in MILLISECONDS
  throughput_images_per_s      test images classified per second
Feature-extraction costs (preprocessing, handcrafted descriptors, AE encoding)
are identical for all classifiers and are reported per image by
train_hybrid.py (outputs/<dataset>/proposed/test_metrics.json -> timing).

Usage
  python classifier_comparison.py --dataset aptos
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from common import LOGGER, base_argparser, dataset_keys, environment_info, load_config, load_json, output_dir, \
    save_json, set_global_seed, setup_logging
from feature_pipeline import CLASSIFIER_LABELS, compose, fit_model, predict, training_matrix
from metrics import _plt, mcnemar, overall_metrics, save_evaluation, evaluate
from train_hybrid import prepare_features


def run(cfg: Dict, key: str) -> pd.DataFrame:
    LOGGER.info("=== Classifier comparison + processing time | dataset: %s ===", key)
    out = output_dir(cfg, key, "classifier_comparison")
    bp_file = Path(cfg["paths"]["output_root"]) / key / "proposed" / "best_params.json"
    bp = load_json(bp_file) if bp_file.exists() else {"pearson_threshold": cfg["feature_selection"]["redundancy_threshold"]}
    tau = bp.pop("pearson_threshold")
    F = prepare_features(cfg, key)
    man, plan, m = F["manifest"], F["plan"], F["masks"]
    y = man["label"].to_numpy()
    Xtr, ytr = training_matrix(F["ae_o"], F["hc_o"], F["ae_a"], F["hc_a"], man, plan, m["train"], "hybrid")
    Xo = compose(F["ae_o"], F["hc_o"], "hybrid")
    Xva, yva, Xte, yte = Xo[m["val"]], y[m["val"]], Xo[m["test"]], y[m["test"]]
    reps = int(cfg["evaluation"].get("timing_repeats", 5))

    rows, preds = [], {}
    for name in cfg["comparison_classifiers"]:
        model, info = fit_model(Xtr, ytr, Xva, yva, cfg, name, True, True, tau, bp if name == "xgboost" else None)
        pred, proba, _ = predict(model, Xte)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter(); model.predict_proba(Xte); times.append(time.perf_counter() - t0)
        t_inf = float(np.median(times))
        preds[name] = pred
        ov = overall_metrics(yte, pred, proba)
        res = evaluate(yte, pred, proba, cfg["evaluation"]["referable_classes"])
        save_evaluation(res, out, name, f"{CLASSIFIER_LABELS[name]} on hybrid features ({key.upper()})")
        rows.append({"classifier": CLASSIFIER_LABELS[name], "key": name,
                     "role": "proposed" if name == "xgboost" else "comparison",
                     **{k: ov.get(k, np.nan) for k in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                                                        "quadratic_weighted_kappa", "macro_specificity",
                                                        "macro_auc_ovr")},
                     "referable_sensitivity": res["clinical"]["referable_sensitivity"],
                     "train_images": int(len(ytr)), "test_images": int(len(yte)),
                     "features_to_classifier": info["n_model_features"],
                     "train_time_s": info["train_time_s"],
                     "inference_ms_per_image": 1000.0 * t_inf / max(len(yte), 1),
                     "throughput_images_per_s": len(yte) / t_inf if t_inf > 0 else np.nan})
        LOGGER.info("  %-22s acc=%.4f macro-F1=%.4f train=%.2fs infer=%.4f ms/img", CLASSIFIER_LABELS[name],
                    ov["accuracy"], ov["macro_f1"], info["train_time_s"], rows[-1]["inference_ms_per_image"])

    tab = pd.DataFrame(rows)
    if "xgboost" in preds:
        tests = {k: mcnemar(yte, preds["xgboost"], p) for k, p in preds.items() if k != "xgboost"}
        tab["mcnemar_p_vs_xgboost"] = tab["key"].map(lambda k: tests[k]["p_value"] if k in tests else np.nan)
        save_json(tests, out / "mcnemar_vs_xgboost.json")
    tab.to_csv(out / "classifier_comparison.csv", index=False)
    save_json(environment_info(), out / "hardware_environment.json")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    col = ["#1d3557" if r == "proposed" else "#9db4c0" for r in tab["role"]]
    axes[0].barh(tab["classifier"], tab["macro_f1"], color=col); axes[0].invert_yaxis()
    axes[0].set_xlabel("Macro-F1 (test set)")
    axes[1].barh(tab["classifier"], tab["inference_ms_per_image"], color=col); axes[1].invert_yaxis()
    axes[1].set_xlabel("Inference time (ms per image)")
    fig.suptitle(f"Classifiers on identical hybrid features — {key.upper()}")
    fig.savefig(out / "classifier_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tab


def main() -> None:
    args = base_argparser("Classifier comparison and processing-time analysis.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "classifier_comparison.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        run(cfg, key)


if __name__ == "__main__":
    main()
