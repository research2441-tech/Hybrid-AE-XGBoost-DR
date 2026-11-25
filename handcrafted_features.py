"""
handcrafted_features.py
-----------------------
Explicitly computed retinal fundus descriptors that complement the
autoencoder representation. All descriptors are computed inside the circular
field of view (FOV) so that the black background does not bias them.

Feature groups (default configuration -> 73 features)
  * Texture (36): Haralick properties of the grey-level co-occurrence matrix
    (GLCM) of the green channel: contrast, dissimilarity, homogeneity,
    energy, correlation and angular second moment (ASM), for distances
    d = 1, 2, 4 pixels; for every distance the mean and the range over the
    four orientations (0, 45, 90, 135 degrees) are kept (rotation-robust).
  * Statistical (16): mean, variance, standard deviation, skewness, kurtosis,
    Shannon entropy, 10th/50th/90th percentiles and inter-quartile range of
    the green channel; mean and standard deviation of the red and blue
    channels; area fraction of bright (exudate-like) and dark
    (haemorrhage/vessel-like) structures obtained with morphological
    top-hat / black-hat filtering.
  * Shape (21): the seven Hu moment invariants of (i) the green-channel
    intensity image, (ii) the bright-structure map and (iii) the
    dark-structure map, log-scaled as sign(h) * log10(|h|).

No descriptor has learned parameters, so extraction is leakage-free and can be
computed once for every image.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

from preprocessing import circular_mask

_CHANNEL = {"red": 0, "green": 1, "blue": 2}


def feature_names(hcfg: Dict) -> List[str]:
    names: List[str] = []
    for d in hcfg["glcm_distances"]:
        for prop in hcfg["glcm_properties"]:
            names += [f"glcm_{prop}_d{d}_mean", f"glcm_{prop}_d{d}_range"]
    names += ["stat_g_mean", "stat_g_var", "stat_g_std", "stat_g_skew", "stat_g_kurtosis", "stat_g_entropy",
              "stat_g_p10", "stat_g_p50", "stat_g_p90", "stat_g_iqr",
              "stat_r_mean", "stat_r_std", "stat_b_mean", "stat_b_std",
              "stat_bright_area_frac", "stat_dark_area_frac"]
    for src in ("intensity", "bright", "dark"):
        names += [f"hu_{src}_{i + 1}" for i in range(7)]
    return names


def _glcm_features(channel: np.ndarray, fov: np.ndarray, hcfg: Dict) -> List[float]:
    levels = int(hcfg["glcm_levels"])
    # Quantise FOV pixels to 1..levels-1; background is level 0 and removed below.
    q = np.zeros(channel.shape, dtype=np.uint8)
    q[fov] = 1 + (channel[fov].astype(np.float32) * (levels - 1) / 256.0).astype(np.uint8)
    angles = [np.deg2rad(a) for a in hcfg["glcm_angles_deg"]]
    dists = list(hcfg["glcm_distances"])
    P = graycomatrix(q, distances=dists, angles=angles, levels=levels, symmetric=True, normed=False)
    P = P.astype(np.float64)
    P[0, :, :, :] = 0.0
    P[:, 0, :, :] = 0.0
    sums = P.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1.0
    P = P / sums
    feats: List[float] = []
    props = {p: graycoprops(P, p) for p in hcfg["glcm_properties"]}   # (n_dist, n_angle)
    for di in range(len(dists)):
        for p in hcfg["glcm_properties"]:
            v = np.nan_to_num(props[p][di])
            feats += [float(v.mean()), float(v.max() - v.min())]
    return feats


def _lesion_maps(green: np.ndarray, fov: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    erode = cv2.erode(fov.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # ignore FOV rim
    bright = np.zeros_like(green)
    dark = np.zeros_like(green)
    if erode.any():
        tb, td = tophat[erode], blackhat[erode]
        bright[erode & (tophat > tb.mean() + 2.5 * tb.std())] = 255
        dark[erode & (blackhat > td.mean() + 2.0 * td.std())] = 255
    return bright, dark, erode


def _hu(img: np.ndarray, log: bool) -> List[float]:
    h = cv2.HuMoments(cv2.moments(img.astype(np.float32))).flatten()
    if log:
        h = -np.sign(h) * np.log10(np.abs(h) + 1e-30)
    return np.nan_to_num(h).tolist()


def extract_features(img: np.ndarray, hcfg: Dict) -> np.ndarray:
    """Return the handcrafted descriptor vector of one preprocessed RGB image."""
    size = img.shape[0]
    fov = circular_mask(size) & (img.max(axis=2) > 0)
    if fov.sum() < 100:
        fov = circular_mask(size)
    ch = img[..., _CHANNEL[hcfg.get("channel", "green")]]
    feats = _glcm_features(ch, fov, hcfg)

    g = ch[fov].astype(np.float64)
    hist, _ = np.histogram(g, bins=64, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    p10, p25, p50, p75, p90 = np.percentile(g, [10, 25, 50, 75, 90])
    r, b = img[..., 0][fov].astype(np.float64), img[..., 2][fov].astype(np.float64)
    bright, dark, inner = _lesion_maps(ch, fov)
    area = max(inner.sum(), 1)
    feats += [g.mean(), g.var(), g.std(), float(np.nan_to_num(skew(g))), float(np.nan_to_num(kurtosis(g))),
              entropy, p10, p50, p90, p75 - p25, r.mean(), r.std(), b.mean(), b.std(),
              float((bright > 0).sum() / area), float((dark > 0).sum() / area)]

    log = bool(hcfg.get("hu_log_transform", True))
    feats += _hu(ch * fov, log) + _hu(bright, log) + _hu(dark, log)
    return np.asarray(feats, dtype=np.float32)


def extract_batch(images: List[np.ndarray] | np.ndarray, hcfg: Dict, n_jobs: int = -1) -> np.ndarray:
    if len(images) == 0:
        return np.zeros((0, len(feature_names(hcfg))), np.float32)
    rows = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(extract_features)(im, hcfg) for im in images)
    return np.vstack(rows).astype(np.float32)
