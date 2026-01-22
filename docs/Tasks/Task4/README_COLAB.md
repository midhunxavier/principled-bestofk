# Task 4.1: TSP Experiments - Colab Execution Guide

This guide provides the 3 code snippets required to run the full Phase 4 TSP experimental grid (TSP50 + TSP100) on Google Colab, utilizing the validated `maxk_pomo` configuration (Hybrid Variance Reduction, λ=0.5, Z-Score norm).

## Setup (Run this first in a code cell)

```python
import os
import sys

# CHANGE THIS to the actual path of the repo in your Drive
os.environ['REPO_DIR'] = "/content/drive/MyDrive/Colab_Notebooks/BestofK_V2"

# Paths
os.environ['PYTHONPATH'] = f"{os.environ['REPO_DIR']}/code"
os.environ['RUNS_DIR'] = "/content/drive/MyDrive/Colab_Notebooks/BestofK_V1/runs_t4_2"
os.environ['ARTIFACTS_DIR'] = "/content/drive/MyDrive/Colab_Notebooks/BestofK_V1/artifacts_t4_2"

# Verify directory
!ls -F "$REPO_DIR/code/src/experiments/"
```

---

## Snippet 1: Train All Algorithms (TSP50 & TSP100)

This runs the training grid for `pomo`, `leader_reward`, and `maxk_pomo` on both TSP50 and TSP100.
**Note:** `maxk_pomo` is configured with `k=8`, `hybrid` variance reduction, and `zscore` normalization.

```bash
%%bash
set -euo pipefail
cd "$REPO_DIR"

python3 -m src.experiments.phase4_tsp_experiments train \
    --runs_dir "$RUNS_DIR" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --num_locs 50 100 \
    --algorithms pomo leader_reward maxk_pomo \
    --seeds 0 1 2 \
    --num_starts 32 \
    --max_epochs 50 \
    --batch_size 128 \
    --train_data_size 20000 \
    --val_data_size 2000 \
    --lr 1e-4 \
    --accelerator gpu \
    --devices 1 \
    --maxk_k 8 \
    --maxk_variance_reduction hybrid \
    --maxk_hybrid_lambda 0.5 \
    --maxk_weight_normalization zscore \
    --maxk_min_gap_scale 0.01 \
    --leader_alpha 0.5
```

---

## Snippet 2: Evaluate All Checkpoints

This evaluates all trained checkpoints using both **Greedy** decoding and **Best-of-128** sampling.

```bash
%%bash
set -euo pipefail
cd "$REPO_DIR"

python3 -m src.experiments.phase4_tsp_experiments eval \
    --runs_dir "$RUNS_DIR" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --num_locs 50 100 \
    --algorithms pomo leader_reward maxk_pomo \
    --seeds 0 1 2 \
    --num_starts 32 \
    --maxk_k 8 \
    --maxk_variance_reduction hybrid \
    --maxk_weight_normalization zscore \
    --maxk_min_gap_scale 0.01 \
    --maxk_hybrid_lambda 0.5 \
    --leader_alpha 0.5 \
    --num_instances 1000 \
    --batch_size 256 \
    --k_eval 128 \
    --device cuda
```

*Note: The arguments passed here (like `maxk_k`, `seeds`) are used to reconstruct the directory names to find the correct checkpoints.*

---

## Snippet 3: Summarize Results (Tables & Plots)

This aggregates the evaluation results into Markdown tables and generates training curves.

```bash
%%bash
set -euo pipefail
cd "$REPO_DIR"

python3 -m src.experiments.phase4_tsp_experiments summarize \
    --runs_dir "$RUNS_DIR" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --num_locs 50 100 \
    --algorithms pomo leader_reward maxk_pomo \
    --seeds 0 1 2 \
    --num_starts 32 \
    --maxk_k 8 \
    --maxk_variance_reduction hybrid \
    --maxk_weight_normalization zscore \
    --maxk_min_gap_scale 0.01 \
    --maxk_hybrid_lambda 0.5 \
    --leader_alpha 0.5 \
    --k_eval 128
```

### Outputs
After running Snippet 3, check the following in your Google Drive:
1.  **Tables:** `$ARTIFACTS_DIR/tables/t4_1_tsp_results.md`
2.  **Plots:** `$ARTIFACTS_DIR/plots/` (Training curves for TSP50 and TSP100)
