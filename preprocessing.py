"""
preprocessing.py
----------------
Retinal fundus image preprocessing and training-only data augmentation.

Preprocessing (deterministic, no learned parameters -> no leakage risk)
  1. Crop the black background surrounding the field of view (FOV).
  2. Resize to a square image (config: preprocessing.image_size).
  3. Apply a circular FOV mask to suppress non-retinal pixels.
  4. Wiener filtering for noise reduction.
  5. CLAHE on the L channel (LAB colour space) for brightness/contrast
     normalisation across acquisition devices.
  6. (Optional) Graham local-average colour subtraction.

Augmentation (applied ONLY to images of the training partition)
  horizontal flip, rotation, brightness scaling and contrast scaling.
  Every synthetic image keeps the identifier of its parent image
  (``parent_id``), which is what the leakage audit in
  ``data_preparation.py`` and ``feature_pipeline.py`` checks.
"""
from __future__ import annotations

import hashlib
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from scipy.signal import wiener

cv2.setNumThreads(1)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_to_fov(img: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Remove the dark border around the circular retinal field of view."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    cropped = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    # Pad to square so that the subsequent resize does not distort the FOV.
    h, w = cropped.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=img.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = cropped
    return out


def circular_mask(size: int, margin: float = 0.98) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    r = margin * size / 2.0
    return ((yy - c) ** 2 + (xx - c) ** 2) <= r ** 2


def wiener_denoise(img: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return img
    f = img.astype(np.float64)
    out = np.empty_like(f)
    for ch in range(3):
        plane = wiener(f[..., ch], (window, window))
        out[..., ch] = np.nan_to_num(plane, nan=0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_lab(img: np.ndarray, clip: float, tiles: int) -> np.ndarray:
    if clip is None or clip <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def ben_graham(img: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def preprocess_image(img: np.ndarray, cfg_pre: Dict) -> np.ndarray:
    """Full deterministic preprocessing chain for one RGB fundus image."""
    size = int(cfg_pre["image_size"])
    img = crop_to_fov(img, int(cfg_pre.get("crop_threshold", 7)))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = wiener_denoise(img, int(cfg_pre.get("wiener_window", 0)))
    img = clahe_lab(img, cfg_pre.get("clahe_clip_limit", 0), cfg_pre.get("clahe_tile_grid", 8))
    img = ben_graham(img, cfg_pre.get("ben_graham_sigma", 0))
    if cfg_pre.get("circular_mask", True):
        img = img * circular_mask(size)[..., None]
    return img.astype(np.uint8)


def _preprocess_worker(args: Tuple[str, str, Dict]) -> Tuple[str, bool, str]:
    src, dst, cfg_pre = args
    try:
        if Path(dst).exists():
            return dst, True, ""
        out = preprocess_image(read_rgb(src), cfg_pre)
        cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return dst, True, ""
    except Exception as exc:  # pragma: no cover - reported to caller
        return dst, False, str(exc)


def config_fingerprint(cfg_pre: Dict) -> str:
    """Short hash so caches are rebuilt whenever preprocessing settings change."""
    text = repr(sorted(cfg_pre.items())).encode()
    return hashlib.md5(text).hexdigest()[:8]


def build_preprocessed_cache(sources: List[str], cache_folder: Path, ids: List[str],
                             cfg_pre: Dict, n_jobs: int = -1) -> Tuple[List[str], List[str]]:
    """Preprocess every source image once and store it as PNG in ``cache_folder``."""
    cache_folder.mkdir(parents=True, exist_ok=True)
    dsts = [str(cache_folder / f"{i}.png") for i in ids]
    jobs = [(s, d, cfg_pre) for s, d in zip(sources, dsts)]
    workers = None if n_jobs in (-1, None) else max(1, int(n_jobs))
    failures: List[str] = []
    if workers == 1 or len(jobs) < 64:
        results = map(_preprocess_worker, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_preprocess_worker, jobs, chunksize=32)
    for dst, ok, msg in results:
        if not ok:
            failures.append(f"{dst}: {msg}")
    return dsts, failures


def load_cached(path: str | Path) -> np.ndarray:
    return read_rgb(path)


# --------------------------------------------------------------------------
# Perceptual hash (near-duplicate detection used by data_preparation.py)
# --------------------------------------------------------------------------
def _retinal_core(img: np.ndarray) -> np.ndarray:
    """Central square inscribed in the circular FOV, so that the black border
    and the disc outline (identical in every fundus photograph) do not dominate
    the hash."""
    size = img.shape[0]
    half = int(size * 0.34)
    c = size // 2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray[c - half:c + half, c - half:c + half]


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.flatten():
        value = (value << 1) | int(b)
    return value


def phash64(img: np.ndarray) -> int:
    """64-bit DCT perceptual hash of the retinal core of an RGB fundus image."""
    small = cv2.resize(_retinal_core(img), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct.flatten()[1:])
    return _bits_to_int(dct > med)


def dhash64(img: np.ndarray) -> int:
    """64-bit difference hash; used to confirm pHash candidates (two independent hashes)."""
    small = cv2.resize(_retinal_core(img), (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


# --------------------------------------------------------------------------
# Training-only augmentation
# --------------------------------------------------------------------------
def _rng_for(parent_id: str, k: int, seed: int) -> np.random.Generator:
    """Deterministic generator per (parent image, copy index, global seed)."""
    key = zlib.crc32(f"{parent_id}|{k}|{seed}".encode())
    return np.random.default_rng(key)


def augment_image(img: np.ndarray, parent_id: str, k: int, cfg_aug: Dict, seed: int) -> Tuple[np.ndarray, Dict]:
    """Return the k-th synthetic copy of ``img`` and the parameters used."""
    rng = _rng_for(parent_id, k, seed)
    out = img.astype(np.float32)
    params: Dict[str, float] = {}
    size = img.shape[0]
    fov = circular_mask(size)[..., None]

    flip = rng.random() < float(cfg_aug.get("hflip_prob", 0.5))
    if flip:
        out = out[:, ::-1]
    params["hflip"] = float(flip)

    deg = float(rng.uniform(-cfg_aug.get("rotation_deg", 0), cfg_aug.get("rotation_deg", 0)))
    if abs(deg) > 1e-6:
        m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), deg, 1.0)
        out = cv2.warpAffine(out, m, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    params["rotation_deg"] = deg

    d = float(cfg_aug.get("brightness_delta", 0.0))
    b = float(rng.uniform(1 - d, 1 + d)) if d > 0 else 1.0
    out = out * b
    params["brightness"] = b

    lo, hi = cfg_aug.get("contrast_range", [1.0, 1.0])
    c = float(rng.uniform(lo, hi))
    mean = out[fov[..., 0]].mean() if fov.any() else out.mean()
    out = (out - mean) * c + mean
    params["contrast"] = c

    out = np.clip(out, 0, 255) * fov
    return out.astype(np.uint8), params


def augmentation_plan(train_labels: np.ndarray, train_ids: Iterable[str], cfg_aug: Dict,
                      seed: int) -> List[Tuple[str, int, int]]:
    """Decide how many synthetic copies each TRAINING image receives.

    Returns a list of (parent_id, copy_index, label). The plan only ever
    references identifiers of the training partition.
    """
    if not cfg_aug.get("enabled", True):
        return []
    ids = np.asarray(list(train_ids))
    labels = np.asarray(train_labels)
    classes, counts = np.unique(labels, return_counts=True)
    cap = int(cfg_aug.get("max_aug_per_image", 8))
    if cfg_aug.get("target_per_class"):
        target = int(cfg_aug["target_per_class"])
    else:
        target = int(counts.max())
    rng = np.random.default_rng(seed)
    plan: List[Tuple[str, int, int]] = []
    for cls, n in zip(classes, counts):
        need = max(0, min(target - n, n * cap))
        if need == 0:
            continue
        members = ids[labels == cls]
        per = need // n
        extra = need % n
        bonus = set(rng.choice(members, size=extra, replace=False).tolist()) if extra else set()
        for pid in members:
            copies = per + (1 if pid in bonus else 0)
            for k in range(copies):
                plan.append((str(pid), k, int(cls)))
    return plan
