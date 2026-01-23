# Phase 4 V3: Head-to-Head Comparative Study (TSP50)

**Date:** January 22, 2026
**Objective:** Benchmarking Max@K against standard baselines (POMO, Leader Reward).

## 1. Overview
Following the successful stabilization of the Max@K algorithm (Hybrid Variance Reduction) on TSP50, the next logical step is a controlled head-to-head comparison. This experiment runs the proposed method alongside the two strongest relevant baselines to quantify the "Best-of-K" advantage.

## 2. Experimental Setup
*   **Problem:** TSP50 (Euclidean, 2D)
*   **Training Samples:** 100,000 (Scaled up from 20k to ensure convergence)
*   **Validation Samples:** 2,000
*   **Batch Size:** 64
*   **Rollouts (N):** 64 (Increased from 32 for better performance)
*   **Epochs:** 100 (Longer training for SOTA convergence)

## 3. Algorithms & Commands

### A. Baseline 1: Standard POMO
*The industry standard baseline. Optimizes for the average case.*

```bash
python3 -m src.experiments.train_tsp \
    --run_name "tsp50_baseline_pomo_n64" \
    --algorithm pomo \
    --num_loc 50 \
    --num_starts 64 \
    --train_data_size 100000 \
    --val_data_size 2000 \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-4 \
    --accelerator gpu \
    --devices 1
```

### B. Baseline 2: Leader Reward
*A heuristic "Best-of-K" baseline used in recent literature (e.g., RL4CO/POMO variants).*

```bash
python3 -m src.experiments.train_tsp \
    --run_name "tsp50_baseline_leader_n64" \
    --algorithm leader_reward \
    --alpha 1.0 \
    --num_loc 50 \
    --num_starts 64 \
    --train_data_size 100000 \
    --val_data_size 2000 \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-4 \
    --accelerator gpu \
    --devices 1
```

### C. Method: Max@K (Hybrid)
*Our principled estimator. Uses `k=16` (Top 25% of 64) and Hybrid blending for stability.*

```bash
python3 -m src.experiments.train_tsp \
    --run_name "tsp50_maxk_hybrid_k16_n64" \
    --algorithm maxk_pomo \
    --k 16 \
    --variance_reduction hybrid \
    --hybrid_lambda 0.5 \
    --weight_normalization zscore \
    --min_gap_scale 0.01 \
    --num_loc 50 \
    --num_starts 64 \
    --train_data_size 100000 \
    --val_data_size 2000 \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-4 \
    --accelerator gpu \
    --devices 1
```

## 4. Analysis Plan
After the runs complete, we will compare:
1.  **Peak Performance:** Best validation reward achieved (Optimality Gap).
2.  **Convergence Speed:** How fast does each method reach a "good" solution (e.g., -6.0)?
3.  **Stability:** Variance in the loss and validation curves.

**Hypothesis:**
*   **POMO** will be the most stable but may plateau at a local optimum.
*   **Leader Reward** might perform better than POMO but could be unstable.
*   **Max@K (Hybrid)** should match POMO's stability (due to hybrid blending) while reaching a better final optimum (due to the Max@K objective).
