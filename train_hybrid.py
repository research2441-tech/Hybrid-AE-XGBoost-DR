"""
train_hybrid.py
---------------
End-to-end training and held-out evaluation of the proposed framework:

  Algorithm 1 (implemented exactly in this order)
   1  load the leakage-audited split manifest (data_preparation.py);
   2  train the convolutional autoencoder on training images, early-stop on
      validation images;
   3  extract AE latent features and handcrafted descriptors for originals and
      for the training-only augmented copies;
   4  hybrid vector = [z || h];
   5  choose the Pearson threshold and XGBoost hyperparameters on the
      VALIDATION partition (macro-F1);
   6  fit imputation -> Pearson filter -> standardisation -> PCA -> XGBoost on
      the TRAINING partition only;
   7  evaluate once on the untouched TEST partition: overall, class-wise and
      clinical metrics, bootstrap 95 % CIs, confusion matrix, ROC curves;
   8  record training time, inference time per image and the hardware used.

Usage
  python train_hybrid.py --dataset aptos
  python train_hybrid.py --dataset all
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from common import (CLASS_NAMES, LOGGER, base_argparser, dataset_keys, environment_info, load_config,
                    output_dir, save_json, set_global_seed, setup_logging)
from data_preparation import load_manifest, load_plan
from feature_pipeline import (autoencoder_bank, fit_model, handcrafted_bank, hybrid_names, predict,
                              training_matrix, tune_on_validation)
from metrics import bootstrap_ci, evaluate, save_evaluation


def masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {s: (manifest["split"] == s).to_numpy() for s in ("train", "val", "test")}


def prepare_features(cfg: Dict, key: str, force: bool = False) -> Dict:
    """Train (or load) the AE and build/load the hybrid feature bank for a dataset."""
    from autoencoder import ae_euclidean_distance, load_models, save_reconstruction_figure, train_autoencoder
    from preprocessing import load_cached

    manifest = load_manifest(cfg, key)
    plan = load_plan(cfg, key)
    m = masks(manifest)
    fdir = output_dir(cfg, key, "features")
    adir = output_dir(cfg, key, "autoencoder")
    bank_file = fdir / "hybrid_features.npz"
    timing: Dict[str, float] = {}

    if (adir / "encoder.keras").exists() and not force:
        encoder, autoenc = load_models(adir)
    else:
        t0 = time.perf_counter()
        encoder, autoenc, _ = train_autoencoder(manifest["cache_path"][m["train"]].tolist(),
                                                manifest["cache_path"][m["val"]].tolist(),
                                                cfg["autoencoder"], adir, int(cfg["seed"]))
        timing["autoencoder_training_s"] = time.perf_counter() - t0
        sample = np.stack([load_cached(p) for p in manifest["cache_path"][m["test"]].iloc[:6]])
        save_reconstruction_figure(autoenc, sample, int(cfg["autoencoder"]["input_size"]),
                                   adir / "reconstruction_examples.png")

    if bank_file.exists() and not force:
        d = np.load(bank_file)
        bank = {k: d[k] for k in d.files}
    else:
        t0 = time.perf_counter()
        hc_o, hc_a = handcrafted_bank(cfg, key, manifest, plan)
        timing["handcrafted_extraction_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ae_o, ae_a = autoencoder_bank(encoder, cfg, manifest, plan)
        timing["ae_encoding_s"] = time.perf_counter() - t0
        paths = manifest["cache_path"].tolist()
        aeed = np.concatenate([ae_euclidean_distance(autoenc, np.stack([load_cached(p) for p in paths[s:s + 512]]),
                                                     int(cfg["autoencoder"]["input_size"]))
                               for s in range(0, len(paths), 512)])
        bank = {"hc_o": hc_o, "hc_a": hc_a, "ae_o": ae_o, "ae_a": ae_a, "ae_ed": aeed}
        np.savez_compressed(bank_file, **bank)
        if timing:
            save_json(timing, fdir / "feature_extraction_timing.json")
    return {"manifest": manifest, "plan": plan, "masks": m, "encoder": encoder, "autoencoder": autoenc, **bank}


def per_image_latency(cfg: Dict, manifest: pd.DataFrame, encoder, model, n: int = 64) -> Dict[str, float]:
    """Median per-image latency (milliseconds) of each inference stage on n test images."""
    from autoencoder import encode
    from handcrafted_features import extract_features
    from preprocessing import preprocess_image, read_rgb

    rows = manifest[manifest["split"] == "test"].head(n)
    t_pre, t_hc, t_ae, t_clf = [], [], [], []
    size = int(cfg["autoencoder"]["input_size"])
    for src in rows["source_path"]:
        t0 = time.perf_counter(); img = preprocess_image(read_rgb(src), cfg["preprocessing"]); t_pre.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); h = extract_features(img, cfg["handcrafted"]); t_hc.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); z = encode(encoder, img[None], size); t_ae.append(time.perf_counter() - t0)
        x = np.hstack([z, h[None]])
        t0 = time.perf_counter(); model.predict_proba(x); t_clf.append(time.perf_counter() - t0)
    ms = lambda v: float(np.median(v) * 1000) if v else float("nan")  # noqa: E731
    out = {"n_images_timed": len(t_pre), "preprocessing_ms_per_image": ms(t_pre),
           "handcrafted_ms_per_image": ms(t_hc), "autoencoder_encoding_ms_per_image": ms(t_ae),
           "classification_ms_per_image": ms(t_clf)}
    out["end_to_end_ms_per_image"] = sum(out[k] for k in out if k.endswith("_ms_per_image"))
    return out


def run(cfg: Dict, key: str) -> Dict:
    LOGGER.info("=== Proposed framework | dataset: %s ===", key)
    out = output_dir(cfg, key, "proposed")
    F = prepare_features(cfg, key)
    man, plan, m = F["manifest"], F["plan"], F["masks"]
    y = man["label"].to_numpy()

    Xtr, ytr = training_matrix(F["ae_o"], F["hc_o"], F["ae_a"], F["hc_a"], man, plan, m["train"], "hybrid")
    Xall = np.hstack([F["ae_o"], F["hc_o"]])
    Xva, yva = Xall[m["val"]], y[m["val"]]
    Xte, yte = Xall[m["test"]], y[m["test"]]
    LOGGER.info("Hybrid feature matrix: train %s (incl. augmented), val %s, test %s",
                Xtr.shape, Xva.shape, Xte.shape)

    best = tune_on_validation(Xtr, ytr, Xva, yva, cfg, out)
    model, info = fit_model(Xtr, ytr, Xva, yva, cfg, threshold=best["threshold"], xgb_overrides=best["xgb"])
    pred, proba, t_inf = predict(model, Xte)

    res = evaluate(yte, pred, proba, cfg["evaluation"]["referable_classes"])
    res["bootstrap_ci"] = bootstrap_ci(yte, pred, proba, cfg["evaluation"]["bootstrap_iterations"],
                                       cfg["evaluation"]["confidence_level"], int(cfg["seed"]))
    res["selected_hyperparameters"] = {"pearson_threshold": best["threshold"], **best["xgb"]}
    res["pipeline_info"] = info
    res["timing"] = {"train_time_s": info["train_time_s"], "test_images": int(len(yte)),
                     "classifier_inference_total_s": t_inf,
                     "classifier_inference_ms_per_image": 1000 * t_inf / max(len(yte), 1),
                     **per_image_latency(cfg, man, F["encoder"], model)}
    res["environment"] = environment_info()
    res["training_samples"] = {"original": int(m["train"].sum()), "augmented": int(len(plan)), "total": int(len(ytr))}
    save_json(res, out / "test_metrics.json")
    save_evaluation(res, out, "test", f"Proposed AE + handcrafted + XGBoost ({key.upper()} test set)", yte, proba)

    pd.DataFrame({"image_id": man["image_id"][m["test"]].to_numpy(), "true": yte, "pred": pred,
                  **{f"p_{c.replace(' ', '_')}": proba[:, i] for i, c in enumerate(CLASS_NAMES)}}) \
        .to_csv(out / "test_predictions.csv", index=False)
    names = np.array(hybrid_names(cfg))
    sel = model.named_steps["pearson"]
    pd.DataFrame({"feature": names, "selected": sel.support_, "relevance_abs_r": sel.relevance_}) \
        .sort_values("relevance_abs_r", ascending=False).to_csv(out / "pearson_feature_selection.csv", index=False)
    if "pca" in model.named_steps:
        ev = model.named_steps["pca"].explained_variance_ratio_
        pd.DataFrame({"component": np.arange(1, len(ev) + 1), "explained_variance_ratio": ev,
                      "cumulative": np.cumsum(ev)}).to_csv(out / "pca_explained_variance.csv", index=False)
    joblib.dump(model, out / "hybrid_xgboost_pipeline.joblib")
    save_json(res["selected_hyperparameters"], out / "best_params.json")

    o = res["overall"]
    LOGGER.info("[%s] TEST  acc=%.4f  bal-acc=%.4f  macro-F1=%.4f  kappa=%.4f  QWK=%.4f",
                key, o["accuracy"], o["balanced_accuracy"], o["macro_f1"], o["cohen_kappa"],
                o["quadratic_weighted_kappa"])
    return res


def main() -> None:
    args = base_argparser("Train and evaluate the proposed hybrid AE + XGBoost framework.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "train_hybrid.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        run(cfg, key)


if __name__ == "__main__":
    main()
