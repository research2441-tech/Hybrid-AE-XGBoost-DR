"""
generate_report.py
------------------
Collects every result file produced by the pipeline into one Markdown report,
outputs/RESULTS_SUMMARY.md, whose tables can be copied directly into the
manuscript. Generating all numbers from the same result files guarantees that
the Abstract, tables, figures and Conclusion report identical values
(Reviewer 1, comment 20; Reviewer 2, comments 8 and 9).

Usage
  python generate_report.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from common import base_argparser, load_config


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df is None or df.empty:
        return "_not available — run the corresponding script first._\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).to_numpy()]
    return "\n".join([head, sep, *body]) + "\n"


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _read_json(p: Path) -> Dict:
    return json.loads(p.read_text()) if p.exists() else {}


def ci(m: Dict, key: str) -> str:
    v = m.get("overall", {}).get(key)
    c = m.get("bootstrap_ci", {}).get(key)
    if v is None:
        return "n/a"
    return f"{v:.4f} [{c[0]:.4f}, {c[1]:.4f}]" if c else f"{v:.4f}"


def section_dataset(root: Path, key: str) -> List[str]:
    rep = _read_json(root / key / "data" / "dataset_report.json")
    stats = _read_csv(root / key / "data" / "dataset_statistics.csv")
    lines = [f"### {key.upper()} — dataset composition and partition\n"]
    if rep:
        lines.append(f"- Split level: **{rep.get('split_level')}**; split groups: {rep.get('n_groups')}; "
                     f"near-duplicate photograph pairs merged into one group: {rep.get('near_duplicate_pairs')} "
                     f"(label conflicts: {rep.get('near_duplicate_label_conflicts')}).")
        lines.append(f"- Usable images: {rep.get('usable_total')} (official: {rep.get('expected_total', 'n/a')}); "
                     f"missing files removed: {rep.get('missing_image_files_removed')}; "
                     f"invalid labels removed: {rep.get('invalid_labels_removed')}.")
        a = rep.get("leakage_audit", {})
        lines.append(f"- Leakage audit passed: **{a.get('passed')}** (groups spanning splits: "
                     f"{a.get('groups_spanning_multiple_splits')}, augmented copies with non-training parent: "
                     f"{a.get('augmented_copies_with_non_training_parent')}).\n")
    if not stats.empty:
        cols = ["class", "original", "train_original", "train_augmented_copies", "train_after_augmentation",
                "validation", "test"]
        lines.append(md_table(stats[cols], 0))
    return lines


def section_proposed(root: Path, key: str) -> List[str]:
    m = _read_json(root / key / "proposed" / "test_metrics.json")
    lines = [f"### {key.upper()} — proposed framework on the held-out test set\n"]
    if not m:
        return lines + [md_table(pd.DataFrame())]
    lines.append("| Metric | Value [95 % bootstrap CI] |\n| --- | --- |")
    for k, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"),
                   ("macro_precision", "Macro precision"), ("macro_recall", "Macro sensitivity"),
                   ("macro_specificity", "Macro specificity"), ("cohen_kappa", "Cohen's kappa"),
                   ("quadratic_weighted_kappa", "Quadratic weighted kappa"), ("mcc", "MCC"),
                   ("macro_auc_ovr", "Macro ROC-AUC (OvR)")]:
        lines.append(f"| {lab} | {ci(m, k)} |")
    lines.append("")
    lines.append("**Class-wise (one-vs-rest)**\n")
    lines.append(md_table(pd.DataFrame(m["classwise"])[["class", "support", "precision", "sensitivity",
                                                         "specificity", "npv", "f1"]]))
    lines.append("**Clinical error analysis**\n")
    lines.append(md_table(pd.DataFrame([m["clinical"]])))
    t = m.get("timing", {})
    lines.append(f"Selected on validation: {m.get('selected_hyperparameters')}; features after Pearson: "
                 f"{m['pipeline_info'].get('n_selected_features')} of {m['pipeline_info'].get('n_input_features')}; "
                 f"PCA components: {m['pipeline_info'].get('pca_components')} "
                 f"({m['pipeline_info'].get('pca_explained_variance', 0):.3f} variance).\n")
    lines.append(f"Timing — training (Pearson + scaler + PCA + XGBoost): {t.get('train_time_s', float('nan')):.2f} s; "
                 f"per-image inference: preprocessing {t.get('preprocessing_ms_per_image', float('nan')):.2f} ms, "
                 f"handcrafted {t.get('handcrafted_ms_per_image', float('nan')):.2f} ms, AE encoding "
                 f"{t.get('autoencoder_encoding_ms_per_image', float('nan')):.2f} ms, XGBoost "
                 f"{t.get('classification_ms_per_image', float('nan')):.3f} ms, end-to-end "
                 f"{t.get('end_to_end_ms_per_image', float('nan')):.2f} ms.\n")
    return lines


def main() -> None:
    args = base_argparser("Assemble all results into outputs/RESULTS_SUMMARY.md").parse_args()
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["paths"]["output_root"])
    keys = [k for k, v in cfg["datasets"].items() if v.get("enabled", True)]
    L: List[str] = ["# Results summary (auto-generated)\n",
                    "All values below are read from result files written by the scripts of this repository. "
                    "Report these values, unchanged, in the Abstract, Results and Conclusion.\n",
                    "## 1. Datasets and leakage-controlled partitions\n"]
    for k in keys:
        L += section_dataset(root, k)
    L.append("## 2. Proposed framework (AE + handcrafted + Pearson + PCA + XGBoost)\n")
    for k in keys:
        L += section_proposed(root, k)
    L.append("## 3. Ablation study\n")
    for k in keys:
        t = _read_csv(root / k / "ablation" / "ablation_results.csv")
        L += [f"### {k.upper()}\n", md_table(t[["code", "variant", "features_to_classifier", "accuracy",
                                                 "balanced_accuracy", "macro_f1", "cohen_kappa",
                                                 "mcnemar_p_vs_full"]] if not t.empty else t)]
    L.append("## 4. Classifier comparison on identical hybrid features, with processing time\n")
    for k in keys:
        t = _read_csv(root / k / "classifier_comparison" / "classifier_comparison.csv")
        L += [f"### {k.upper()}\n", md_table(t[["classifier", "role", "accuracy", "macro_f1", "cohen_kappa",
                                                 "train_time_s", "inference_ms_per_image",
                                                 "mcnemar_p_vs_xgboost"]] if not t.empty else t)]
        hw = _read_json(root / k / "classifier_comparison" / "hardware_environment.json")
        if hw:
            L.append(f"Hardware: {hw.get('cpu_model', hw.get('processor'))}, {hw.get('cpu_count_logical')} logical "
                     f"cores, {hw.get('ram_gb')} GB RAM, GPU: {hw.get('gpu_model', hw.get('gpus') or 'none')}; "
                     f"Python {hw.get('python')}, XGBoost {hw.get('xgboost_version')}, scikit-learn "
                     f"{hw.get('sklearn_version')}, TensorFlow {hw.get('tensorflow_version')}.\n")
    L.append("## 5. CNN baselines trained on the identical partitions\n")
    for k in keys:
        t = _read_csv(root / k / "baselines" / "baseline_comparison.csv")
        L += [f"### {k.upper()}\n", md_table(t)]
    L.append("## 6. Repeated stratified group cross-validation (mean ± SD, 95 % CI)\n")
    for k in keys:
        t = _read_csv(root / k / "repeated_cv" / "cv_summary.csv")
        if not t.empty:
            show = pd.DataFrame({"classifier": t["classifier"]})
            for mm in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "quadratic_weighted_kappa"):
                show[mm] = [f"{a:.4f} ± {b:.4f} [{c:.4f}, {d:.4f}]" for a, b, c, d in
                            zip(t[f"{mm}_mean"], t[f"{mm}_sd"], t[f"{mm}_ci_low"], t[f"{mm}_ci_high"])]
            t = show
        L += [f"### {k.upper()}\n", md_table(t)]
        sig = _read_json(root / k / "repeated_cv" / "cv_significance.json")
        if "macro_f1" in sig:
            L.append(f"Macro-F1 significance (vs XGBoost): Friedman {sig['macro_f1'].get('friedman')}; "
                     f"Holm-adjusted Wilcoxon {sig['macro_f1'].get('wilcoxon_holm')}.\n")
    L.append("## 7. Cross-dataset generalisation\n")
    t = _read_csv(root / "cross_dataset" / "cross_dataset_results.csv")
    L.append(md_table(t[["source", "target", "evaluation_set", "n_images", "accuracy", "macro_f1",
                         "quadratic_weighted_kappa", "within_source_macro_f1", "generalisation_gap_macro_f1",
                         "aeed_source_median", "aeed_target_median"]] if not t.empty else t))
    L.append("## 8. Complete configuration used\n")
    L.append("```yaml\n" + yaml.safe_dump(cfg, sort_keys=False) + "```\n")
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {root / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
