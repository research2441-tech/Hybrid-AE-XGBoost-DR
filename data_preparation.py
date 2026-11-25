"""
data_preparation.py
-------------------
Builds a leakage-controlled, fully documented experimental partition for each
retinal fundus dataset (Reviewer 1, comments 3, 4, 5 and 6).

Steps
  1. Read the official label file, validate grades (0-4) and image files, and
     compare class counts with the official dataset counts.
  2. Preprocess every ORIGINAL image once (preprocessing.py) and cache it.
  3. Build split groups:
       * Kaggle/EyePACS -> patient level (left and right eye of one patient
         share a group, parsed from the file name);
       * APTOS 2019    -> image level (no patient identifier is distributed).
     In both cases near-duplicate photographs detected with a 64-bit
     perceptual hash are merged into the same group, so that a photograph and
     its duplicate can never fall on both sides of the split.
  4. Stratified group partition into train / validation / test using a fixed
     seed (StratifiedGroupKFold). This happens BEFORE augmentation.
  5. Compute the training-only augmentation plan and report original,
     augmented, training, validation and test counts per class with totals
     that are verified arithmetically.
  6. Run a leakage audit and write all tables to outputs/<dataset>/data/.

Usage
  python data_preparation.py --dataset aptos
  python data_preparation.py --dataset all
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from common import (CLASS_NAMES, LOGGER, base_argparser, cache_dir, dataset_keys, load_config,
                    output_dir, save_json, set_global_seed, setup_logging)
from preprocessing import (augmentation_plan, build_preprocessed_cache, config_fingerprint, dhash64,
                           load_cached, phash64)


# --------------------------------------------------------------------------
# Label loading
# --------------------------------------------------------------------------
def load_labels(dcfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(dcfg["label_csv"])
    id_col, lab_col = dcfg["id_column"], dcfg["label_column"]
    df = df[[id_col, lab_col]].rename(columns={id_col: "image_id", lab_col: "label"})
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.(png|jpe?g|tif+)$", "", regex=True)
    report: Dict = {"rows_in_label_file": int(len(df))}

    dup_ids = int(df["image_id"].duplicated().sum())
    df = df.drop_duplicates("image_id")
    report["duplicate_ids_removed"] = dup_ids

    bad = ~df["label"].isin(range(len(CLASS_NAMES)))
    report["invalid_labels_removed"] = int(bad.sum())
    df = df[~bad].copy()
    df["label"] = df["label"].astype(int)

    img_dir = Path(dcfg["image_dir"])
    ext = dcfg.get("image_ext", ".png")
    df["source_path"] = [str(img_dir / f"{i}{ext}") for i in df["image_id"]]
    exists = df["source_path"].map(lambda p: Path(p).exists())
    report["missing_image_files_removed"] = int((~exists).sum())
    df = df[exists].reset_index(drop=True)

    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    report["usable_counts"] = counts.tolist()
    report["usable_total"] = int(counts.sum())
    exp = dcfg.get("expected_counts")
    if exp:
        report["expected_counts"] = list(map(int, exp))
        report["expected_total"] = int(sum(exp))
        report["matches_official_counts"] = bool(list(counts) == list(exp))
        if not report["matches_official_counts"]:
            LOGGER.warning("Class counts %s differ from the official counts %s. The difference is "
                           "documented in dataset_report.json.", counts.tolist(), exp)
    return df, report


# --------------------------------------------------------------------------
# Grouping (patient level + near-duplicate merging)
# --------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    count = np.zeros(x.shape, dtype=np.int64)
    for shift in range(0, 64, 8):
        byte = ((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.int64)
        count += _POPCOUNT8[byte]
    return count


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, bands: int) -> List[Tuple[int, int, int]]:
    """Locality-sensitive search for pHash pairs with Hamming distance <= threshold.

    The 64-bit hash is cut into ``bands`` equal bands; two hashes within
    distance < bands share at least one identical band (pigeonhole principle),
    so candidate generation has full recall for threshold <= bands - 1.
    """
    h = hashes.astype(np.uint64)
    width = 64 // bands
    mask = np.uint64((1 << width) - 1)
    seen = set()
    pairs: List[Tuple[int, int, int]] = []
    for b in range(bands):
        keys = (h >> np.uint64(b * width)) & mask
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        boundaries = np.flatnonzero(np.diff(sk)) + 1
        for bucket in np.split(order, boundaries):
            if len(bucket) < 2:
                continue
            sub = h[bucket]
            for start in range(0, len(bucket), 1024):          # row chunks bound memory use
                block = sub[start:start + 1024]
                dist = _popcount64(block[:, None] ^ sub[None, :])
                ii, jj = np.where(dist <= threshold)
                for a, c in zip(ii, jj):
                    i, j = int(bucket[start + a]), int(bucket[c])
                    if i == j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((key[0], key[1], int(dist[a, c])))
    return pairs


def build_groups(df: pd.DataFrame, dcfg: Dict, scfg: Dict) -> Tuple[np.ndarray, Dict]:
    n = len(df)
    uf = UnionFind(n)
    info: Dict = {}

    regex = dcfg.get("patient_id_regex")
    if regex:
        pat = re.compile(regex)
        pids = [m.group(1) if (m := pat.match(i)) else i for i in df["image_id"]]
        df["patient_id"] = pids
        first: Dict[str, int] = {}
        for idx, p in enumerate(pids):
            if p in first:
                uf.union(first[p], idx)
            else:
                first[p] = idx
        info["split_level"] = "patient"
        info["n_patients"] = len(first)
    else:
        df["patient_id"] = df["image_id"]
        info["split_level"] = "image"

    info["near_duplicate_pairs"] = 0
    info["near_duplicate_label_conflicts"] = 0
    if scfg.get("duplicate_grouping", True):
        thr = int(scfg.get("phash_hamming_threshold", 6))
        cand = near_duplicate_pairs(df["phash"].to_numpy(np.uint64), thr, int(scfg.get("phash_bands", 8)))
        dh = df["dhash"].to_numpy(np.uint64)
        # A pair is a near duplicate only if BOTH independent hashes agree.
        pairs = [(i, j, d) for i, j, d in cand
                 if int(_popcount64(np.array([dh[i] ^ dh[j]], dtype=np.uint64))[0]) <= thr]
        info["phash_candidate_pairs"] = len(cand)
        labels = df["label"].to_numpy()
        conflicts = 0
        for i, j, _ in pairs:
            uf.union(i, j)
            conflicts += int(labels[i] != labels[j])
        info["near_duplicate_pairs"] = len(pairs)
        info["near_duplicate_label_conflicts"] = conflicts
        pd.DataFrame([(df["image_id"][i], df["image_id"][j], d, labels[i], labels[j]) for i, j, d in pairs],
                     columns=["image_a", "image_b", "hamming", "label_a", "label_b"]) \
            .to_csv(Path(dcfg["_out"]) / "near_duplicate_pairs.csv", index=False)

    roots = np.array([uf.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    info["n_groups"] = int(groups.max() + 1)
    info["largest_group_size"] = int(np.bincount(groups).max())
    if info["largest_group_size"] > 20:
        LOGGER.warning("Largest split group has %d images; inspect near_duplicate_pairs.csv and consider a lower "
                       "split.phash_hamming_threshold.", info["largest_group_size"])
    return groups, info


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------
def stratified_group_partition(y: np.ndarray, groups: np.ndarray, scfg: Dict, seed: int) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' labels (one per image)."""
    n = len(y)
    split = np.array(["train"] * n, dtype=object)
    test_frac, val_frac = float(scfg["test_fraction"]), float(scfg["val_fraction"])
    idx = np.arange(n)
    if scfg.get("stratify", True):
        k1 = max(2, int(round(1.0 / test_frac)))
        sgkf = StratifiedGroupKFold(n_splits=k1, shuffle=True, random_state=seed)
        dev_idx, test_idx = next(sgkf.split(idx, y, groups))
        split[test_idx] = "test"
        if val_frac > 0:
            k2 = max(2, int(round((1.0 - test_frac) / val_frac)))
            sgkf2 = StratifiedGroupKFold(n_splits=k2, shuffle=True, random_state=seed + 1)
            tr_rel, va_rel = next(sgkf2.split(dev_idx, y[dev_idx], groups[dev_idx]))
            split[dev_idx[va_rel]] = "val"
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        dev_idx, test_idx = next(gss.split(idx, y, groups))
        split[test_idx] = "test"
        if val_frac > 0:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed + 1)
            _, va_rel = next(gss2.split(dev_idx, y[dev_idx], groups[dev_idx]))
            split[dev_idx[va_rel]] = "val"
    return split


def leakage_audit(manifest: pd.DataFrame, plan: List[Tuple[str, int, int]]) -> Dict:
    """Verify that no group, patient or augmented copy crosses partitions."""
    audit: Dict = {}
    g = manifest.groupby("group")["split"].nunique()
    audit["groups_spanning_multiple_splits"] = int((g > 1).sum())
    p = manifest.groupby("patient_id")["split"].nunique()
    audit["patients_spanning_multiple_splits"] = int((p > 1).sum())
    split_of = dict(zip(manifest["image_id"], manifest["split"]))
    audit["augmented_copies_total"] = len(plan)
    audit["augmented_copies_with_non_training_parent"] = int(sum(split_of[pid] != "train" for pid, _, _ in plan))
    ids = manifest["image_id"]
    audit["image_ids_in_multiple_splits"] = int(ids.duplicated().sum())
    audit["passed"] = all(audit[k] == 0 for k in ("groups_spanning_multiple_splits",
                                                    "patients_spanning_multiple_splits",
                                                    "augmented_copies_with_non_training_parent",
                                                    "image_ids_in_multiple_splits"))
    return audit


def statistics_table(manifest: pd.DataFrame, plan: List[Tuple[str, int, int]]) -> pd.DataFrame:
    """Class-wise counts whose row and column totals are checked arithmetically."""
    rows = []
    aug_counts = pd.Series([lab for _, _, lab in plan], dtype=int).value_counts()
    for c, name in enumerate(CLASS_NAMES):
        m = manifest[manifest["label"] == c]
        tr, va, te = (m["split"] == "train").sum(), (m["split"] == "val").sum(), (m["split"] == "test").sum()
        aug = int(aug_counts.get(c, 0))
        rows.append({"class_id": c, "class": name, "original": len(m), "train_original": tr,
                     "train_augmented_copies": aug, "train_after_augmentation": tr + aug,
                     "validation": va, "test": te})
    tab = pd.DataFrame(rows)
    total = tab.drop(columns=["class_id", "class"]).sum()
    tab.loc[len(tab)] = {"class_id": -1, "class": "Total", **total.to_dict()}
    body = tab.iloc[:-1]
    assert (body["train_original"] + body["validation"] + body["test"] == body["original"]).all()
    assert (body["train_original"] + body["train_augmented_copies"] == body["train_after_augmentation"]).all()
    assert int(tab.iloc[-1]["original"]) == len(manifest)
    for col in ("train_original", "validation", "test"):
        tab[f"{col}_pct"] = (100.0 * tab[col] / tab.iloc[-1]["original"]).round(2)
    return tab


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def prepare_dataset(cfg: Dict, key: str) -> pd.DataFrame:
    dcfg = dict(cfg["datasets"][key])
    out = output_dir(cfg, key, "data")
    dcfg["_out"] = str(out)
    seed = int(cfg["seed"])
    LOGGER.info("[%s] reading labels from %s", key, dcfg["label_csv"])
    df, report = load_labels(dcfg)

    fp = config_fingerprint(cfg["preprocessing"])
    folder = cache_dir(cfg, key, f"preprocessed_{fp}")
    LOGGER.info("[%s] preprocessing %d images -> %s", key, len(df), folder)
    paths, failures = build_preprocessed_cache(df["source_path"].tolist(), folder, df["image_id"].tolist(),
                                               cfg["preprocessing"], cfg.get("n_jobs", -1))
    df["cache_path"] = paths
    if failures:
        LOGGER.warning("[%s] %d images failed preprocessing and are excluded", key, len(failures))
        bad = {Path(f.split(":")[0]).stem for f in failures}
        df = df[~df["image_id"].isin(bad)].reset_index(drop=True)
    report["preprocessing_failures"] = len(failures)

    LOGGER.info("[%s] computing perceptual (pHash) and difference (dHash) hashes", key)
    hashes = [(phash64(im), dhash64(im)) for im in map(load_cached, df["cache_path"])]
    df["phash"] = [np.uint64(a) for a, _ in hashes]
    df["dhash"] = [np.uint64(b) for _, b in hashes]

    groups, ginfo = build_groups(df, dcfg, cfg["split"])
    df["group"] = groups
    report.update(ginfo)

    df["split"] = stratified_group_partition(df["label"].to_numpy(), groups, cfg["split"], seed)
    train = df[df["split"] == "train"]
    plan = augmentation_plan(train["label"].to_numpy(), train["image_id"], cfg["augmentation"], seed)
    audit = leakage_audit(df, plan)
    if not audit["passed"]:
        raise RuntimeError(f"Leakage audit failed: {audit}")

    stats = statistics_table(df, plan)
    manifest = df[["image_id", "patient_id", "group", "label", "split", "source_path", "cache_path"]].copy()
    manifest["phash"] = [format(int(h), "016x") for h in df["phash"]]
    manifest["dhash"] = [format(int(h), "016x") for h in df["dhash"]]
    manifest.to_csv(out / "split_manifest.csv", index=False)
    pd.DataFrame(plan, columns=["parent_id", "copy_index", "label"]).to_csv(out / "augmentation_plan.csv", index=False)
    stats.to_csv(out / "dataset_statistics.csv", index=False)

    report.update({"dataset": key, "seed": seed, "split_config": cfg["split"],
                   "augmentation_config": cfg["augmentation"], "leakage_audit": audit,
                   "preprocessing_fingerprint": fp})
    save_json(report, out / "dataset_report.json")
    LOGGER.info("[%s] split level=%s | groups=%d | near-duplicate pairs=%d | audit passed=%s",
                key, report["split_level"], report["n_groups"], report["near_duplicate_pairs"], audit["passed"])
    LOGGER.info("[%s] dataset statistics:\n%s", key,
                stats[["class", "original", "train_original", "train_augmented_copies",
                       "train_after_augmentation", "validation", "test"]].to_string(index=False))
    return manifest


def load_manifest(cfg: Dict, key: str) -> pd.DataFrame:
    path = Path(cfg["paths"]["output_root"]) / key / "data" / "split_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python data_preparation.py --dataset {key}")
    return pd.read_csv(path, dtype={"image_id": str, "patient_id": str})


def load_plan(cfg: Dict, key: str) -> List[Tuple[str, int, int]]:
    path = Path(cfg["paths"]["output_root"]) / key / "data" / "augmentation_plan.csv"
    if not path.exists():
        return []
    p = pd.read_csv(path, dtype={"parent_id": str})
    return list(zip(p["parent_id"], p["copy_index"].astype(int), p["label"].astype(int)))


def main() -> None:
    args = base_argparser(__doc__.split("\n")[1]).parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs" / "data_preparation.log")
    set_global_seed(int(cfg["seed"]))
    for key in dataset_keys(cfg, args.dataset):
        prepare_dataset(cfg, key)


if __name__ == "__main__":
    main()
