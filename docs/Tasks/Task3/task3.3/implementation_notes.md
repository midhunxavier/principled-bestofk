# Task 3.3 — Implementation Notes (Stability mechanisms)

**Task:** T3.3 — Implement stability mechanisms (clipping, normalization)  
**Status:** Implemented  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- Training script uses Lightning gradient clipping (norm):
  - `code/src/experiments/train_tsp.py`
    - passes `gradient_clip_algorithm="norm"` and exposes `--gradient_clip_val`
- Optional (experimental) scaling / normalization:
  - `code/src/experiments/train_tsp.py`
    - `--reward_scale {norm,scale,<int>}`
  - `code/src/algorithms/maxk_pomo.py`
    - applies RL4CO’s `RewardScaler` (`self.advantage_scaler`) to Max@K weights before the loss
  - `code/src/algorithms/leader_reward.py`
    - continues to use RL4CO’s `RewardScaler` on advantages (inherited behavior)
- Numerical stability / debugging guards (opt-in):
  - `--check_numerics` enables `torch.isfinite(...)` checks in:
    - `src.algorithms.maxk_pomo.MaxKPOMO.calculate_loss`
    - `src.algorithms.leader_reward.LeaderRewardPOMO.calculate_loss`
  - `--debug_clamp_weights <float>` clamps weights/advantages after scaling (debug-only; biases gradients)

---

## 2. Notes on correctness vs. stability

- By default, **no extra scaling/normalization** is applied beyond the unbiased estimator + (optional) LOO variance reduction.
- `--reward_scale <int>` behaves like a step-size change (constant division).
- `--reward_scale {norm,scale}` uses running statistics and is **experimental**: it can change the effective update rule (and should be treated as a stability knob, not a “principled” estimator component).
- `--debug_clamp_weights` is strictly for debugging/extreme-case stabilization and should be kept off for principled comparisons.

---

## 3. Usage

Examples:

- Enable NaN/inf checks:
  - `python3 code/src/experiments/train_tsp.py --algorithm maxk_pomo --num_loc 20 --k 4 --check_numerics --max_epochs 1`
- Normalize weights/advantages (experimental):
  - `python3 code/src/experiments/train_tsp.py --algorithm maxk_pomo --num_loc 20 --k 4 --reward_scale norm --max_epochs 1`
- Debug clamp (biases gradients; use only to diagnose explosions):
  - `python3 code/src/experiments/train_tsp.py --algorithm maxk_pomo --num_loc 20 --k 4 --debug_clamp_weights 1.0 --check_numerics --max_epochs 1`
