"""
smoke_test.py
-------------
Self-contained functional test of the complete repository that needs no
download. It synthesises two small fundus-like image collections that mimic
the file layouts of APTOS 2019 (hexadecimal ids, includes duplicated
photographs) and Kaggle/EyePACS ("<patient>_left/right"), then executes every
stage through its command-line interface with a reduced configuration:

  data_preparation -> train_hybrid -> ablation -> classifier_comparison
  -> repeated_cv -> baselines_cnn -> cross_dataset -> generate_report

and asserts that the leakage audit passes and that all result files exist.
The synthetic images only verify that the code runs correctly end to end; the
numbers obtained on them have no clinical meaning.

Usage
  python smoke_test.py            (about 5-15 minutes on a laptop CPU)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path("smoke_workspace")


def synth_fundus(grade: int, rng: np.random.Generator, size: int = 320) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    r = int(size * rng.uniform(0.40, 0.46))
    base = np.array([rng.uniform(150, 210), rng.uniform(60, 100), rng.uniform(20, 45)])
    cv2.circle(img, (c, c), r, tuple(int(v) for v in base), -1)
    od = (int(c + rng.uniform(-0.45, 0.45) * r), int(c + rng.uniform(-0.15, 0.15) * r))
    cv2.circle(img, od, int(r * 0.13), (240, 220, 160), -1)
    for _ in range(7):                                       # vessel tree
        ang = rng.uniform(0, 2 * np.pi)
        pts = [od]
        for s in range(1, 6):
            a = ang + rng.normal(0, 0.25)
            pts.append((int(od[0] + np.cos(a) * r * 0.18 * s), int(od[1] + np.sin(a) * r * 0.18 * s)))
        cv2.polylines(img, [np.array(pts)], False, (110, 20, 15), int(rng.integers(2, 4)))
    def inside():
        while True:
            x, y = rng.integers(c - r, c + r, 2)
            if (x - c) ** 2 + (y - c) ** 2 < (0.85 * r) ** 2:
                return int(x), int(y)
    for _ in range([0, 10, 22, 34, 40][grade] + int(rng.integers(0, 4))):    # microaneurysms
        cv2.circle(img, inside(), int(rng.integers(2, 4)), (70, 5, 5), -1)
    for _ in range([0, 0, 5, 14, 16][grade]):                                # blot haemorrhages
        cv2.circle(img, inside(), int(rng.integers(5, 11)), (95, 8, 6), -1)
    for _ in range([0, 0, 8, 14, 14][grade]):                                # hard exudates
        cv2.circle(img, inside(), int(rng.integers(3, 6)), (245, 225, 90), -1)
    if grade == 4:                                                           # neovascularisation
        for _ in range(8):
            p = inside()
            pts = np.array([(p[0] + int(8 * np.sin(t)) + int(t * 3), p[1] + int(t * 2)) for t in np.linspace(0, 6, 25)])
            cv2.polylines(img, [pts], False, (150, 25, 20), 2)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    img *= rng.uniform(0.75, 1.2)
    img += rng.normal(0, 4, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_datasets(per_class: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    a_dir = ROOT / "data/aptos2019/train_images"; a_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in range(5):
        for i in range(per_class):
            iid = f"{rng.integers(0, 2**48):012x}"
            cv2.imwrite(str(a_dir / f"{iid}.png"), synth_fundus(g, rng))
            rows.append((iid, g))
    for iid, g in rows[:3]:                                # exact duplicates under new ids
        dup = f"dup{iid[:9]}"
        shutil.copy(a_dir / f"{iid}.png", a_dir / f"{dup}.png")
        rows.append((dup, g))
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(ROOT / "data/aptos2019/train.csv", index=False)

    k_dir = ROOT / "data/kaggle_eyepacs/train"; k_dir.mkdir(parents=True, exist_ok=True)
    rows, pid = [], 10
    for g in range(5):
        for i in range(per_class // 2):
            for eye in ("left", "right"):
                gg = g if eye == "left" or rng.random() < 0.7 else max(0, g - 1)
                cv2.imwrite(str(k_dir / f"{pid}_{eye}.jpeg"), synth_fundus(gg, rng))
                rows.append((f"{pid}_{eye}", gg))
            pid += 1
    pd.DataFrame(rows, columns=["image", "level"]).to_csv(ROOT / "data/kaggle_eyepacs/trainLabels.csv", index=False)


def smoke_config() -> Path:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    cfg["paths"] = {"output_root": str(ROOT / "outputs"), "cache_root": str(ROOT / "cache")}
    cfg["n_jobs"] = 2
    for k, sub in (("aptos", "aptos2019"), ("kaggle", "kaggle_eyepacs")):
        d = cfg["datasets"][k]
        d["label_csv"] = str(ROOT / "data" / sub / Path(d["label_csv"]).name)
        d["image_dir"] = str(ROOT / "data" / sub / Path(d["image_dir"]).name)
        d["expected_counts"] = None
    cfg["augmentation"]["max_aug_per_image"] = 1
    cfg["autoencoder"].update({"input_size": 64, "filters": [16, 32, 64, 128], "latent_dim": 32, "epochs": 3,
                               "batch_size": 32, "early_stopping_patience": 2})
    cfg["feature_selection"]["threshold_grid"] = [0.90, 0.95]
    cfg["xgboost"].update({"n_estimators": 60, "early_stopping_rounds": 10,
                           "tuning_grid": {"max_depth": [3, 6], "learning_rate": [0.1]}})
    cfg["evaluation"].update({"bootstrap_iterations": 100, "timing_repeats": 2})
    cfg["repeated_cv"].update({"n_splits": 3, "n_repeats": 1, "autoencoder_epochs": 2,
                               "classifiers": ["logistic_regression", "random_forest", "xgboost"]})
    cfg["baselines"].update({"models": ["MobileNetV2"], "weights": None, "image_size": 96, "head_epochs": 1,
                             "finetune_epochs": 1, "batch_size": 16})
    p = ROOT / "smoke_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def run(script: str, cfg: Path, *extra: str) -> None:
    cmd = [sys.executable, script, "--config", str(cfg), *extra]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    make_datasets()
    cfg = smoke_config()
    run("data_preparation.py", cfg)
    run("train_hybrid.py", cfg)
    run("ablation.py", cfg)
    run("classifier_comparison.py", cfg)
    run("repeated_cv.py", cfg, "--dataset", "aptos")
    run("baselines_cnn.py", cfg, "--dataset", "aptos")
    run("cross_dataset.py", cfg)
    run("generate_report.py", cfg)

    out = ROOT / "outputs"
    for key in ("aptos", "kaggle"):
        rep = json.loads((out / key / "data" / "dataset_report.json").read_text())
        assert rep["leakage_audit"]["passed"], rep["leakage_audit"]
        for f in ("proposed/test_metrics.json", "proposed/test_confusion_matrix.png", "ablation/ablation_results.csv",
                  "classifier_comparison/classifier_comparison.csv"):
            assert (out / key / f).exists(), f"missing {key}/{f}"
    a = json.loads((out / "aptos" / "data" / "dataset_report.json").read_text())
    assert a["near_duplicate_pairs"] >= 3, "duplicate photographs were not detected"
    k = json.loads((out / "kaggle" / "data" / "dataset_report.json").read_text())
    assert k["split_level"] == "patient"
    for f in ("aptos/repeated_cv/cv_summary.csv", "aptos/baselines/baseline_comparison.csv",
              "cross_dataset/cross_dataset_results.csv", "RESULTS_SUMMARY.md"):
        assert (out / f).exists(), f"missing {f}"
    print("\nSMOKE TEST PASSED — all stages executed and the leakage audit passed for both datasets.")


if __name__ == "__main__":
    main()
