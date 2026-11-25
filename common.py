"""
common.py
---------
Shared utilities for the Hybrid Autoencoder + Handcrafted Features + XGBoost
diabetic retinopathy (DR) grading framework.

Responsibilities
* Configuration loading with dotted command-line overrides
  (e.g. ``--set autoencoder.epochs=5``).
* Global seeding of Python, NumPy and TensorFlow for reproducible runs.
* Output-directory management and JSON/CSV persistence helpers.
* Hardware / software environment capture, so every timing figure that is
  reported can be tied to the exact machine and library versions used
  (Reviewer 1, comment 14).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
NUM_CLASSES = len(CLASS_NAMES)

LOGGER = logging.getLogger("dr_hybrid")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console (and optional file) logging once per process."""
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        LOGGER.addHandler(fh)
    LOGGER.propagate = False
    return LOGGER


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def _parse_scalar(text: str) -> Any:
    """Parse a command-line override value with YAML semantics."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path = "config.yaml",
                overrides: Optional[Iterable[str]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the YAML configuration and apply ``key.sub=value`` overrides."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if extra:
        cfg = deep_update(cfg, extra)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, val = item.split("=", 1)
        _set_dotted(cfg, key.strip(), _parse_scalar(val.strip()))
    return cfg


def base_argparser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by every executable script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    p.add_argument("--dataset", default=None,
                   help="Dataset key defined under `datasets:` in the config (e.g. aptos, kaggle).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Override any config entry, e.g. --set xgboost.max_depth=8")
    return p


def dataset_keys(cfg: Dict[str, Any], requested: Optional[str]) -> list[str]:
    """Resolve which datasets a script should process."""
    enabled = [k for k, v in cfg["datasets"].items() if v.get("enabled", True)]
    if requested is None or requested == "all":
        return enabled
    if requested not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset '{requested}'. Available: {list(cfg['datasets'])}")
    return [requested]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow deterministically."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - older TF builds
            pass
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Paths and persistence
# --------------------------------------------------------------------------
def output_dir(cfg: Dict[str, Any], *parts: str) -> Path:
    """Return (and create) a sub-directory of the configured output root."""
    p = Path(cfg["paths"]["output_root"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(cfg: Dict[str, Any], *parts: str) -> Path:
    p = Path(cfg["paths"]["cache_root"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Timing and environment capture
# --------------------------------------------------------------------------
@contextmanager
def timer(store: Dict[str, float], key: str):
    """Context manager that writes wall-clock seconds into ``store[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0


def environment_info() -> Dict[str, Any]:
    """Hardware and library versions recorded alongside every experiment."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 2)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:  # pragma: no cover
        pass
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "tensorflow", "cv2", "skimage", "scipy"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[f"{mod}_version"] = "not installed"
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]
        if gpus:
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                info["gpu_model"] = details.get("device_name", "unknown")
            except Exception:
                pass
    except ImportError:
        info["gpus"] = []
    return info


def fmt_mean_sd(values: Iterable[float], digits: int = 4) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return "n/a"
    sd = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.{digits}f} ± {sd:.{digits}f}"
