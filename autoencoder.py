"""
autoencoder.py
--------------
Convolutional autoencoder (AE) used for unsupervised representation learning
from retinal fundus images.

Architecture (default config, input 128 x 128 x 3)
  Encoder: 4 blocks, each = Conv2D(f, 3x3, stride 1) -> BN -> LeakyReLU(0.2)
                           -> Conv2D(f, 3x3, stride 2) -> BN -> LeakyReLU(0.2)
           with f = 32, 64, 128, 256  ->  8 x 8 x 256
           Flatten -> Dense(latent_dim = 256, linear)  = latent vector z
  Decoder: Dense(8*8*256) -> LeakyReLU -> Reshape(8, 8, 256)
           4 x [Conv2DTranspose(f, 3x3, stride 2) -> BN -> LeakyReLU(0.2)]
           Conv2D(3, 3x3) -> sigmoid  (reconstruction in [0, 1])
  Loss:    L = MSE(x, x_hat) + w * (1 - SSIM(x, x_hat)),  w = 0.2
  Optimiser: Adam (lr 1e-3, ReduceLROnPlateau, early stopping on validation loss)
  Regularisation: L2 (1e-5) on all kernels, dropout (0.2) on z while training.

The AE is fitted on TRAINING images only; validation images are used for early
stopping. The encoder output z is the learned part of the hybrid feature vector.

The module also implements the AE Euclidean distance (AE-ED):
  AE-ED(x) = || x - g(f(x)) ||_2
which is used in cross_dataset.py as a quantitative indicator of the
distribution shift between source and target fundus image collections.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from common import LOGGER, save_json


# --------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------
def _act(name: str):
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.2)
    return layers.Activation(name)


def build_autoencoder(acfg: Dict) -> Tuple[keras.Model, keras.Model]:
    size = int(acfg["input_size"])
    filters: List[int] = list(acfg["filters"])
    k = int(acfg["kernel_size"])
    reg = regularizers.l2(float(acfg.get("l2", 0.0))) if acfg.get("l2") else None
    bn = bool(acfg.get("batch_norm", True))
    act = acfg.get("activation", "leaky_relu")

    inp = keras.Input((size, size, 3), name="fundus_image")
    x = inp
    for i, f in enumerate(filters):
        for stride in (1, 2):
            x = layers.Conv2D(f, k, strides=stride, padding="same", kernel_regularizer=reg,
                              name=f"enc{i + 1}_conv_s{stride}")(x)
            if bn:
                x = layers.BatchNormalization(name=f"enc{i + 1}_bn_s{stride}")(x)
            x = _act(act)(x)
    spatial = size // (2 ** len(filters))
    x = layers.Flatten(name="enc_flatten")(x)
    z = layers.Dense(int(acfg["latent_dim"]), kernel_regularizer=reg, name="latent")(x)
    encoder = keras.Model(inp, z, name="encoder")

    zin = keras.Input((int(acfg["latent_dim"]),), name="latent_in")
    y = layers.Dropout(float(acfg.get("dropout", 0.0)), name="latent_dropout")(zin)
    y = layers.Dense(spatial * spatial * filters[-1], kernel_regularizer=reg, name="dec_dense")(y)
    y = _act(act)(y)
    y = layers.Reshape((spatial, spatial, filters[-1]), name="dec_reshape")(y)
    for i, f in enumerate(reversed(filters)):
        y = layers.Conv2DTranspose(f, k, strides=2, padding="same", kernel_regularizer=reg,
                                   name=f"dec{i + 1}_deconv")(y)
        if bn:
            y = layers.BatchNormalization(name=f"dec{i + 1}_bn")(y)
        y = _act(act)(y)
    out = layers.Conv2D(3, k, padding="same", activation="sigmoid", name="reconstruction")(y)
    decoder = keras.Model(zin, out, name="decoder")

    autoencoder = keras.Model(inp, decoder(encoder(inp)), name="autoencoder")
    return encoder, autoencoder


def make_loss(acfg: Dict):
    w = float(acfg.get("ssim_weight", 0.2))
    if acfg.get("loss", "mse") == "mse_ssim":
        def mse_ssim(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
            ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
            return mse + w * (1.0 - ssim)
        return mse_ssim
    return "mse"


# --------------------------------------------------------------------------
# Input pipelines
# --------------------------------------------------------------------------
def _path_dataset(paths: Sequence[str], size: int, batch: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(p):
        img = tf.io.decode_png(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, (size, size), method="area") / 255.0
        return img

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.map(lambda x: tf.image.random_flip_left_right(x, seed=seed), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def to_ae_input(images: np.ndarray, size: int) -> np.ndarray:
    """uint8 (N, H, W, 3) -> float32 (N, size, size, 3) in [0, 1]."""
    t = tf.image.resize(tf.convert_to_tensor(images), (size, size), method="area")
    return (t.numpy() / 255.0).astype(np.float32)


# --------------------------------------------------------------------------
# Training / inference
# --------------------------------------------------------------------------
def train_autoencoder(train_paths: Sequence[str], val_paths: Sequence[str], acfg: Dict,
                      out_dir: Path, seed: int, epochs: int | None = None) -> Tuple[keras.Model, keras.Model, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)
    encoder, autoencoder = build_autoencoder(acfg)
    opt = keras.optimizers.Adam(learning_rate=float(acfg["learning_rate"]))
    autoencoder.compile(optimizer=opt, loss=make_loss(acfg), metrics=["mse"])

    size, bs = int(acfg["input_size"]), int(acfg["batch_size"])
    tr = _path_dataset(train_paths, size, bs, True, seed)
    va = _path_dataset(val_paths, size, bs, False, seed)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(acfg["early_stopping_patience"]),
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=int(acfg["reduce_lr_patience"]),
                                          min_lr=float(acfg["min_learning_rate"])),
        keras.callbacks.CSVLogger(str(out_dir / "autoencoder_history.csv")),
    ]
    n_ep = int(epochs if epochs is not None else acfg["epochs"])
    LOGGER.info("Training autoencoder: %d train / %d val images, %d epochs max", len(train_paths), len(val_paths), n_ep)
    hist = autoencoder.fit(tr, validation_data=va, epochs=n_ep, callbacks=cbs, verbose=2)

    encoder.save(out_dir / "encoder.keras")
    autoencoder.save(out_dir / "autoencoder.keras")
    lines: List[str] = []
    encoder.summary(print_fn=lines.append)
    autoencoder.get_layer("decoder").summary(print_fn=lines.append)
    (out_dir / "autoencoder_architecture.txt").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "epochs_run": len(hist.history["loss"]),
        "best_val_loss": float(np.min(hist.history["val_loss"])),
        "final_train_loss": float(hist.history["loss"][-1]),
        "encoder_params": int(encoder.count_params()),
        "autoencoder_params": int(autoencoder.count_params()),
        "latent_dim": int(acfg["latent_dim"]),
        "config": acfg,
    }
    save_json(summary, out_dir / "autoencoder_summary.json")
    return encoder, autoencoder, summary


def encode(encoder: keras.Model, images: np.ndarray, size: int, batch: int = 128) -> np.ndarray:
    """Latent features for uint8 images (N, H, W, 3)."""
    feats = []
    for s in range(0, len(images), batch):
        x = to_ae_input(images[s:s + batch], size)
        feats.append(encoder(x, training=False).numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, encoder.output_shape[-1]), np.float32)


def ae_euclidean_distance(autoencoder: keras.Model, images: np.ndarray, size: int, batch: int = 128) -> np.ndarray:
    """AE-ED(x) = ||x - x_hat||_2 computed on the normalised AE input."""
    out = []
    for s in range(0, len(images), batch):
        x = to_ae_input(images[s:s + batch], size)
        xr = autoencoder(x, training=False).numpy()
        out.append(np.sqrt(((x - xr) ** 2).reshape(len(x), -1).sum(axis=1)))
    return np.concatenate(out) if out else np.zeros(0)


def load_models(out_dir: Path) -> Tuple[keras.Model, keras.Model]:
    custom = {}
    enc = keras.models.load_model(out_dir / "encoder.keras", compile=False)
    ae = keras.models.load_model(out_dir / "autoencoder.keras", compile=False, custom_objects=custom)
    return enc, ae


def save_reconstruction_figure(autoencoder: keras.Model, images: np.ndarray, size: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(6, len(images))
    if n == 0:
        return
    x = to_ae_input(images[:n], size)
    xr = autoencoder(x, training=False).numpy()
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(x[i]); axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(xr[i], 0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_title("Input", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstruction", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def history_frame(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "autoencoder_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
