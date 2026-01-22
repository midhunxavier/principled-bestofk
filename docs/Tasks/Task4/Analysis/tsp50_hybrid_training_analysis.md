# TSP50 Max@K Hybrid Training Analysis

**Date:** January 22, 2026
**Algorithm:** Max@K POMO (Hybrid Variance Reduction)
**Task:** TSP50

## 1. Executive Summary
The training run for TSP50 using the `maxk_pomo` algorithm with **Hybrid Variance Reduction** was successful. Unlike previous instability observed in smaller instances (10 locations), this configuration demonstrated stable convergence, robust generalization, and healthy gradient dynamics. The model reached a validation reward of **-6.18**, which is approximately **8-9%** from the known optimal value (~5.70) for TSP50, a strong result for a prototype run with limited rollouts (`num_starts=32`).

## 2. Experiment Configuration
The following hyperparameters were used for this successful run:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Algorithm** | `maxk_pomo` | Principled Max@K Policy Gradient |
| **Problem Size** | TSP50 | 50 locations |
| **Num Starts (N)** | 32 | Rollouts per instance |
| **K** | 8 | Optimizing for top 25% of solutions |
| **Variance Reduction** | `hybrid` | **Critical Factor** |
| **Hybrid Lambda** | 0.5 | 50% SubLOO (Max@K), 50% POMO (Avg) |
| **Normalization** | `zscore` | Zero-mean, unit-variance weights |
| **Training Data** | 20,000 | |
| **Batch Size** | 128 | |
| **Learning Rate** | 1e-4 | Constant (Adam) |

## 3. Training Dynamics Analysis

### 3.1 Convergence Profile
The training showed a classic, healthy reinforcement learning curve:
*   **Initial State (Epoch 0):** Reward ~ `-8.54` (Random policy performance).
*   **Rapid Improvement (Epochs 0-10):** Reward improved quickly to `~-6.56`.
*   **Fine-tuning (Epochs 10-49):** Steady, monotonic improvement to `-6.23` (Train) / `-6.18` (Val).

### 3.2 Loss Interpretation
*   **Observed Loss:** Fluctuated between `-30` and `-136`.
*   **Analysis:** Negative loss values are expected and correct for this implementation. The loss is calculated as $-\sum (w \cdot \log p)$.
    *   Stable negative values indicate the model is consistently assigning high probability ($\log p \approx -20$) to trajectories with positive advantages ($w > 0$).
    *   The absence of divergence (loss exploding to `NaN` or positive infinity) confirms the **z-score normalization** and **hybrid blending** successfully stabilized the gradients.

### 3.3 Generalization
*   The gap between `train/reward` (`-6.23`) and `val/reward` (`-6.18`) is negligible (and actually slightly better on validation, likely due to batch noise).
*   **Conclusion:** The model is learning the underlying TSP heuristic structure rather than overfitting to the 20k training samples.

## 4. Discussion: Why This Succeeded
The user noted previous failures with TSP10. The success of this TSP50 run can be attributed to three key architectural choices:

1.  **Hybrid Variance Reduction (`lambda=0.5`):**
    *   Pure Max@K gradients can be high-variance, especially early in training when the "top K" trajectories are random.
    *   By blending in 50% of the **POMO baseline** (which optimizes for the *average* case and is known to be extremely stable), the model had a safety rail. It learned "generally good" moves from POMO while the Max@K signal pushed it to prefer "peak" performance.

2.  **K/N Ratio ($8/32 = 0.25$):**
    *   Optimizing for the top 25% of trajectories provides a dense, smooth gradient signal.
    *   In previous failures (e.g., if $K=1$), the signal is too sparse (only 1 in 32 trajectories provides positive reinforcement).

3.  **Z-Score Normalization:**
    *   Max@K weights and POMO advantages have different scales. Z-scoring ensures they contribute equally to the gradient magnitude, preventing one objective from dominating the other or causing exploding gradients.

## 5. Recommendations for Next Steps
To close the gap to optimality (~5.70) and reach State-of-the-Art (SOTA) performance:

1.  **Scale Up `num_starts`:**
    *   **Action:** Increase `num_starts` from `32` to **`128`**.
    *   **Reasoning:** Max@K relies on the tail distribution. More samples = better "best" found = stronger Max@K signal.

2.  **Implement Learning Rate Decay:**
    *   **Action:** Decay LR by factor of 0.1 at Epoch 40.
    *   **Reasoning:** The logs show improvement slowing down after Epoch 30. A lower LR will allow the model to settle into a sharper minimum.

3.  **Increase Max@K Focus:**
    *   **Action:** Increase `hybrid_lambda` from `0.5` to **`0.8`** or **`0.9`**.
    *   **Reasoning:** Now that stability is proven, we can bias the model more heavily towards the "Best-of-K" objective to aggressively optimize peak performance.

4.  **Dataset Size:**
    *   **Action:** Increase training data to **100,000** samples.
    *   **Reasoning:** Ensures the model sees a wider variety of graph topologies, improving robustness for larger N.
