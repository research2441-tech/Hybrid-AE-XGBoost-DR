"""
baselines_cnn.py
----------------
Controlled comparison with contemporary deep convolutional networks
(Reviewer 1, comment 9): ResNet50, DenseNet121, EfficientNetB0 and MobileNetV2.

Each baseline is trained and evaluated on EXACTLY the same partition manifest
as the proposed framework:
  * the same preprocessed images (preprocessing.py) at 224 x 224;
  * the same training-only augmented copies (augmentation_plan.csv);
  * the validation partition for early stopping / checkpoint selection;
  * the untouched test partition for the final, single evaluation.

Training protocol (identical for all networks)
  stage 1: ImageNet backbone frozen, new head = GAP -> Dropout(0.3) -> Dense(5, softmax),
           Adam 1e-3, `head_epochs`
  stage 2: top `unfreeze_fraction` of backbone layers unfrozen (BatchNorm kept frozen),
           Adam 1e-4, up to `finetune_epochs`, early stopping on validation loss
  loss: categorical cross-entropy; batch size 32.

Outputs per model: class-wise metrics, confusion matrix, ROC curves, bootstrap
CIs, McNemar's test against the proposed model on the same test images,
parameter count, training time and inference time per image.

Usage
  python baselines_cnn.py --dataset aptos
  python baselines_cnn.py --dataset aptos --set "baselines.models=[MobileNetV2]"
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import (LOGGER, NUM_CLASSES, base_argparser, dataset_keys, environment_info, load_config,
                    output_dir, save_json, set_global_seed, setup_logging)
from data_preparation import load_manifest, load_plan
from metrics import _plt, bootstrap_ci, evaluate, mcnemar, save_evaluation
from preprocessing import augment_image, load_cached

APPS = {
    "ResNet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "DenseNet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input),
    "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    "MobileNetV2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
}


def build_model(name: str, bcfg: Dict) -> Tuple[keras.Model, keras.Model, str]:
    ctor, _ = APPS[name]
    size = int(bcfg["image_size"])
    weights = bcfg.get("weights")
    try:
        base = ctor(include_top=False, weights=weights, input_shape=(size, size, 3))
        used = str(weights)
    except Exception as exc:  # offline machines: fall back to random initialisation, and record it
        LOGGER.warning("Could not load %s weights for %s (%s); using random initialisation.", weights, name, exc)
        base = ctor(include_top=False, weights=None, input_shape=(size, size, 3))
        used = "random (weights unavailable)"
    inp = keras.Input((size, size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(float(bcfg["dropout"]))(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name=name), base, used


def make_dataset(items: List[Tuple], path_of: Dict[str, str], cfg: Dict, prep, shuffle: bool, seed: int):
    """items: (image_id, copy_index or -1, label). Augmented items are rebuilt deterministically."""
    bs = int(cfg["baselines"]["batch_size"])
    size = int(cfg["baselines"]["image_size"])

    def gen():
        order = np.random.default_rng(seed).permutation(len(items)) if shuffle else np.arange(len(items))
        for i in order:
            iid, k, lab = items[i]
            img = load_cached(path_of[iid])
            if k >= 0:
                img = augment_image(img, iid, k, cfg["augmentation"], int(cfg["seed"]))[0]
            if img.shape[0] != size:
                img = tf.image.resize(img, (size, size)).numpy()
            yield img.astype(np.float32), np.eye(NUM_CLASSES, dtype=np.float32)[lab]

    sig = (tf.TensorSpec((size, size, 3), tf.float32), tf.TensorSpec((NUM_CLASSES,), tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    ds = ds.map(lambda x, y: (prep(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


def run_model(name: str, cfg: Dict, key: str, man: pd.DataFrame, plan, out: Path) -> Dict:
    bcfg = cfg["baselines"]
    seed = int(cfg["seed"])
    tf.keras.utils.set_random_seed(seed)
    _, prep = APPS[name]
    path_of = dict(zip(man["image_id"], man["cache_path"]))
    tr = [(i, -1, int(l)) for i, l in zip(man["image_id"][man.split == "train"], man["label"][man.split == "train"])]
    if bcfg.get("use_training_augmentation", True):
        tr += [(p, int(k), int(l)) for p, k, l in plan]
    va = [(i, -1, int(l)) for i, l in zip(man["image_id"][man.split == "val"], man["label"][man.split == "val"])]
    te_df = man[man.split == "test"]
    te = [(i, -1, int(l)) for i, l in zip(te_df["image_id"], te_df["label"])]

    ds_tr = make_dataset(tr, path_of, cfg, prep, True, seed)
    ds_va = make_dataset(va, path_of, cfg, prep, False, seed)
    ds_te = make_dataset(te, path_of, cfg, prep, False, seed)
    labels = np.array([l for _, _, l in tr])
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weight = {c: float(len(labels) / (NUM_CLASSES * max(counts[c], 1))) for c in range(NUM_CLASSES)}

    model, base, used_weights = build_model(name, bcfg)
    mdir = out / name
    mdir.mkdir(parents=True, exist_ok=True)
    ckpt = str(mdir / "best.weights.h5")
    cbs = lambda pat: [keras.callbacks.EarlyStopping(monitor="val_loss", patience=pat, restore_best_weights=True),  # noqa: E731
                       keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True,
                                                       save_weights_only=True),
                       keras.callbacks.CSVLogger(str(mdir / "history.csv"), append=True)]
    t0 = time.perf_counter()
    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(float(bcfg["head_learning_rate"])),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_tr, validation_data=ds_va, epochs=int(bcfg["head_epochs"]), class_weight=class_weight,
              callbacks=cbs(int(bcfg["early_stopping_patience"])), verbose=2)
    base.trainable = True
    n_freeze = int(len(base.layers) * (1 - float(bcfg["unfreeze_fraction"])))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= n_freeze and not isinstance(layer, layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(float(bcfg["finetune_learning_rate"])),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_tr, validation_data=ds_va, epochs=int(bcfg["finetune_epochs"]), class_weight=class_weight,
              callbacks=cbs(int(bcfg["early_stopping_patience"])), verbose=2)
    if Path(ckpt).exists():
        model.load_weights(ckpt)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict(ds_te, verbose=0)
    infer_s = time.perf_counter() - t0
    yte = te_df["label"].to_numpy()
    pred = proba.argmax(1)
    res = evaluate(yte, pred, proba, cfg["evaluation"]["referable_classes"])
    res["bootstrap_ci"] = bootstrap_ci(yte, pred, proba, min(1000, cfg["evaluation"]["bootstrap_iterations"]),
                                       cfg["evaluation"]["confidence_level"], seed)
    res["timing"] = {"train_time_s": train_s, "inference_ms_per_image": 1000 * infer_s / max(len(yte), 1),
                     "test_images": int(len(yte)), "train_images_per_epoch": int(len(tr))}
    res["parameters"] = int(model.count_params())
    res["initial_weights"] = used_weights
    save_evaluation(res, mdir, "test", f"{name} ({key.upper()} test set)", yte, proba)
    pd.DataFrame({"image_id": te_df["image_id"].to_numpy(), "true": yte, "pred": pred}) \
        .to_csv(mdir / "test_predictions.csv", index=False)

    prop = Path(cfg["paths"]["output_root"]) / key / "proposed" / "test_predictions.csv"
    if prop.exists():
        pp = pd.read_csv(prop, dtype={"image_id": str}).set_index("image_id").loc[te_df["image_id"]]
        res["mcnemar_vs_proposed"] = mcnemar(yte, pp["pred"].to_numpy(), pred)
    save_json(res, mdir / "test_metrics.json")
    keras.backend.clear_session()
    return res


def run(cfg: Dict, key: str) -> pd.DataFrame:
    LOGGER.info("=== CNN baselines | dataset: %s ===", key)
    out = output_dir(cfg, key, "baselines")
    man = load_manifest(cfg, key)
    plan = load_plan(cfg, key)
    rows = []
    for name in cfg["baselines"]["models"]:
        LOGGER.info("--- %s ---", name)
        r = run_model(name, cfg, key, man, plan, out)
        o = r["overall"]
        rows.append({"model": name, "parameters": r["parameters"], "initial_weights": r["initial_weights"],
                     **{k: o.get(k, np.nan) for k in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                                                       "quadratic_weighted_kappa", "macro_auc_ovr")},
                     "macro_f1_ci_low": r["bootstrap_ci"]["macro_f1"][0],
                     "macro_f1_ci_high": r["bootstrap_ci"]["macro_f1"][1],
                     "referable_sensitivity": r["clinical"]["referable_sensitivity"],
                     "train_time_s": r["timing"]["train_time_s"],
                     "inference_ms_per_image": r["timing"]["inference_ms_per_image"],
                     "mcnemar_p_vs_proposed": r.get("mcnemar_vs_proposed", {}).get("p_value", np.nan)})
        pd.DataFrame(rows).to_csv(out / "baseline_comparison.csv", index=False)
    tab = pd.DataFrame(rows)
    save_json(environment_info(), out / "hardware_environment.json")

    prop = Path(cfg["paths"]["output_root"]) / key / "proposed" / "test_metrics.json"
    if prop.exists():
        import json
        pr = json.loads(prop.read_text())
        plt = _plt()
        names = list(tab["model"]) + ["Proposed"]
        vals = list(tab["macro_f1"]) + [pr["overall"]["macro_f1"]]
        lo = list(tab["macro_f1_ci_low"]) + [pr["bootstrap_ci"]["macro_f1"][0]]
        hi = list(tab["macro_f1_ci_high"]) + [pr["bootstrap_ci"]["macro_f1"][1]]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(names, vals, xerr=[np.subtract(vals, lo), np.subtract(hi, vals)], capsize=3,
                color=["#9db4c0"] * len(tab) + ["#1d3557"])
        ax.invert_yaxis(); ax.set_xlabel("Macro-F1 on the identical test set (95 % CI)")
        ax.set_title(f"Proposed framework vs CNN baselines — {key.upper()}")
        fig.savefig(out / "baseline_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return tab


def main() -> None:
    args = base_argparser("Train CNN baselines on the identical partitions.").parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "baselines_cnn.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        run(cfg, key)


if __name__ == "__main__":
    main()
