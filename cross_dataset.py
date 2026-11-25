"""
cross_dataset.py
----------------
Cross-dataset and external generalisation (Reviewer 1, comment 10).

Protocol
  * The complete model of the SOURCE dataset (its autoencoder, Pearson filter,
    scaler, PCA and XGBoost, all fitted on the source training partition) is
    applied without any modification or re-fitting to the TARGET images.
  * No target label is used for fitting, feature selection, PCA or tuning.
  * Evaluated on (a) the target test partition and (b) all target images.
  * The within-dataset result of the source model is reported next to the
    cross-dataset result so that the generalisation gap is explicit.
  * Distribution shift is quantified with the autoencoder Euclidean distance
    AE-ED(x) = ||x - g(f(x))||_2 of the SOURCE autoencoder, comparing source
    test images with target images (Mann-Whitney U test).
  * Optional external dataset (e.g. IDRiD or Messidor-2 with ICDR grades)
    through `cross_dataset.external` with an explicit label map; grades that
    cannot be mapped are excluded and counted, never merged silently.

Usage
  python cross_dataset.py                 # APTOS -> Kaggle and Kaggle -> APTOS
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common import (LOGGER, base_argparser, cache_dir, load_config, output_dir, save_json, set_global_seed,
                    setup_logging)
from data_preparation import load_manifest
from feature_pipeline import handcrafted_bank, predict
from metrics import bootstrap_ci, evaluate, save_evaluation
from preprocessing import build_preprocessed_cache, config_fingerprint, load_cached


def _features(cfg: Dict, key: str, man: pd.DataFrame, encoder) -> np.ndarray:
    from autoencoder import encode
    hc_o, _ = handcrafted_bank(cfg, key, man, [])
    size = int(cfg["autoencoder"]["input_size"])
    ae = np.vstack([encode(encoder, np.stack([load_cached(p) for p in man["cache_path"].iloc[s:s + 512]]), size)
                    for s in range(0, len(man), 512)])
    return np.hstack([ae, hc_o])


def _aeed(autoenc, man: pd.DataFrame, size: int, limit: int = 2000, seed: int = 0) -> np.ndarray:
    from autoencoder import ae_euclidean_distance
    if len(man) > limit:
        man = man.sample(limit, random_state=seed)
    paths = man["cache_path"].tolist()
    return np.concatenate([ae_euclidean_distance(autoenc, np.stack([load_cached(p) for p in paths[s:s + 256]]), size)
                           for s in range(0, len(paths), 256)])


def _external_manifest(cfg: Dict) -> pd.DataFrame:
    e = cfg["cross_dataset"]["external"]
    df = pd.read_csv(e["label_csv"])
    df = df[[e["id_column"], e["label_column"]]].rename(columns={e["id_column"]: "image_id", e["label_column"]: "label"})
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.(png|jpe?g|tif+)$", "", regex=True)
    n0 = len(df)
    if e.get("label_map"):
        lm = {int(k): int(v) for k, v in e["label_map"].items()}
        df["label"] = df["label"].map(lm)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(0, 4)]
    df["source_path"] = [str(Path(e["image_dir"]) / f"{i}{e.get('image_ext', '.jpg')}") for i in df["image_id"]]
    df = df[df["source_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    folder = cache_dir(cfg, e["name"], f"preprocessed_{config_fingerprint(cfg['preprocessing'])}")
    df["cache_path"], _ = build_preprocessed_cache(df["source_path"].tolist(), folder, df["image_id"].tolist(),
                                                   cfg["preprocessing"], cfg.get("n_jobs", -1))
    df["split"] = "test"
    LOGGER.info("External dataset %s: %d usable images (%d excluded by label mapping / missing files)",
                e["name"], len(df), n0 - len(df))
    return df


def evaluate_transfer(cfg: Dict, src: str, tgt_name: str, tgt_man: pd.DataFrame, out: Path) -> List[Dict]:
    from autoencoder import load_models
    root = Path(cfg["paths"]["output_root"])
    model = joblib.load(root / src / "proposed" / "hybrid_xgboost_pipeline.joblib")
    encoder, autoenc = load_models(root / src / "autoencoder")
    X = _features(cfg, tgt_name, tgt_man, encoder)
    y = tgt_man["label"].to_numpy()
    rows = []
    subsets = {"target_test_partition": (tgt_man["split"] == "test").to_numpy(),
               "all_target_images": np.ones(len(tgt_man), dtype=bool)}
    within = json.loads((root / src / "proposed" / "test_metrics.json").read_text())["overall"]
    for sub, mask in subsets.items():
        if mask.sum() == 0:
            continue
        pred, proba, _ = predict(model, X[mask])
        res = evaluate(y[mask], pred, proba, cfg["evaluation"]["referable_classes"])
        res["bootstrap_ci"] = bootstrap_ci(y[mask], pred, proba, min(1000, cfg["evaluation"]["bootstrap_iterations"]),
                                           cfg["evaluation"]["confidence_level"], int(cfg["seed"]))
        tag = f"{src}_to_{tgt_name}_{sub}"
        save_json(res, out / f"{tag}.json")
        save_evaluation(res, out, tag, f"Trained on {src.upper()} -> tested on {tgt_name.upper()} ({sub})",
                        y[mask], proba)
        o = res["overall"]
        sens = {f"sensitivity_{r['class'].replace(' ', '_')}": r["sensitivity"] for r in res["classwise"]}
        rows.append({"source": src, "target": tgt_name, "evaluation_set": sub, "n_images": int(mask.sum()),
                     "accuracy": o["accuracy"], "balanced_accuracy": o["balanced_accuracy"],
                     "macro_f1": o["macro_f1"], "quadratic_weighted_kappa": o["quadratic_weighted_kappa"],
                     "macro_f1_ci_low": res["bootstrap_ci"]["macro_f1"][0],
                     "macro_f1_ci_high": res["bootstrap_ci"]["macro_f1"][1],
                     "within_source_macro_f1": within["macro_f1"],
                     "generalisation_gap_macro_f1": within["macro_f1"] - o["macro_f1"],
                     "referable_sensitivity": res["clinical"]["referable_sensitivity"], **sens})
        LOGGER.info("  %s -> %s [%s]: acc=%.4f macro-F1=%.4f (within-source macro-F1=%.4f)", src, tgt_name, sub,
                    o["accuracy"], o["macro_f1"], within["macro_f1"])

    size = int(cfg["autoencoder"]["input_size"])
    src_man = load_manifest(cfg, src)
    d_src = _aeed(autoenc, src_man[src_man["split"] == "test"], size, seed=int(cfg["seed"]))
    d_tgt = _aeed(autoenc, tgt_man, size, seed=int(cfg["seed"]))
    u = mannwhitneyu(d_src, d_tgt, alternative="two-sided")
    shift = {"source": src, "target": tgt_name, "aeed_source_test_median": float(np.median(d_src)),
             "aeed_target_median": float(np.median(d_tgt)), "mann_whitney_u": float(u.statistic),
             "p_value": float(u.pvalue)}
    save_json(shift, out / f"{src}_to_{tgt_name}_aeed_shift.json")
    for r in rows:
        r.update({"aeed_source_median": shift["aeed_source_test_median"], "aeed_target_median": shift["aeed_target_median"]})
    return rows


def main() -> None:
    args = base_argparser("Cross-dataset and external validation.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "cross_dataset.log")
    set_global_seed(int(cfg["seed"]))
    out = output_dir(cfg, "cross_dataset")
    rows: List[Dict] = []
    for src, tgt in cfg["cross_dataset"]["pairs"]:
        LOGGER.info("=== Cross-dataset: %s -> %s ===", src, tgt)
        rows += evaluate_transfer(cfg, src, tgt, load_manifest(cfg, tgt), out)
    ext = cfg["cross_dataset"].get("external", {})
    if ext.get("enabled"):
        man = _external_manifest(cfg)
        for src in {p[0] for p in cfg["cross_dataset"]["pairs"]}:
            LOGGER.info("=== External: %s -> %s ===", src, ext["name"])
            rows += evaluate_transfer(cfg, src, ext["name"], man, out)
    pd.DataFrame(rows).to_csv(out / "cross_dataset_results.csv", index=False)


if __name__ == "__main__":
    main()
