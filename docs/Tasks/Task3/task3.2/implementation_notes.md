# Task 3.2 — Implementation Notes (Training scripts)

**Task:** T3.2 — Integrate with POMO training loop (training script)  
**Status:** Implemented (TSP training entrypoint)  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- Training entrypoint: `code/src/experiments/train_tsp.py`
  - Supports `--algorithm {pomo,maxk_pomo,leader_reward}`
  - Exposes `n := num_starts`, `k`, variance reduction mode, and core Trainer knobs
- Leader Reward algorithm implementation (needed for `--algorithm leader_reward`):
  - `code/src/algorithms/leader_reward.py` (`LeaderRewardPOMO`)

---

## 2. Usage

Run directly from repo root:

- Baseline POMO:
  - `python3 code/src/experiments/train_tsp.py --algorithm pomo --num_loc 20 --max_epochs 1`
- Principled Max@K (SubLOO):
  - `python3 code/src/experiments/train_tsp.py --algorithm maxk_pomo --num_loc 20 --k 4 --variance_reduction subloo --max_epochs 1`
- Leader Reward:
  - `python3 code/src/experiments/train_tsp.py --algorithm leader_reward --num_loc 20 --alpha 0.5 --max_epochs 1`

Artifacts (CSV logs + checkpoints) go under `--output_dir` (default: `.tmp/runs/tsp/`).

---

## 3. Implementation notes

- The script uses RL4CO’s `RL4COTrainer` and logs `train/loss`, `train/reward`, and `val/reward` by default.
- Checkpoints monitor `val/reward` (maximize).
- To avoid cache write issues in constrained environments, the script sets `MPLCONFIGDIR` and `XDG_CACHE_HOME` into `.tmp/` if not already set.

