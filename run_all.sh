#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Reproduces every experiment reported in the manuscript, in dependency order.
#   bash run_all.sh            -> both datasets (APTOS 2019 and Kaggle/EyePACS)
#   bash run_all.sh aptos      -> a single dataset (cross-dataset step skipped)
# All results are written to outputs/ and summarised in outputs/RESULTS_SUMMARY.md
# -----------------------------------------------------------------------------
set -euo pipefail
DATASET="${1:-all}"
CONFIG="${CONFIG:-config.yaml}"

echo "[1/8] Leakage-controlled partitions, dataset statistics and audit"
python data_preparation.py   --config "$CONFIG" --dataset "$DATASET"

echo "[2/8] Proposed framework: AE + handcrafted + Pearson + PCA + XGBoost"
python train_hybrid.py       --config "$CONFIG" --dataset "$DATASET"

echo "[3/8] Ablation study"
python ablation.py           --config "$CONFIG" --dataset "$DATASET"

echo "[4/8] Classifier comparison and processing-time analysis"
python classifier_comparison.py --config "$CONFIG" --dataset "$DATASET"

echo "[5/8] Repeated stratified group cross-validation and significance tests"
python repeated_cv.py        --config "$CONFIG" --dataset "$DATASET"

echo "[6/8] CNN baselines on identical partitions"
python baselines_cnn.py      --config "$CONFIG" --dataset "$DATASET"

if [ "$DATASET" = "all" ]; then
  echo "[7/8] Cross-dataset generalisation"
  python cross_dataset.py    --config "$CONFIG"
else
  echo "[7/8] Cross-dataset generalisation skipped (needs both datasets)"
fi

echo "[8/8] Results summary"
python generate_report.py    --config "$CONFIG"
echo "Done. See outputs/RESULTS_SUMMARY.md"
