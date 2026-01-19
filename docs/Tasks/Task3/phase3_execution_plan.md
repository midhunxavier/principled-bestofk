# Task 3 — Phase 3 Execution Plan: RL4CO Integration (Week 5–6)

**Phase:** 3 (RL4CO Integration)  
**Scope:** T3.1–T3.5 from `docs/PRD.md`  
**Status:** In progress (T3.1–T3.4 implemented)  
**Owner:** Research Team  
**Last updated:** 2026-01-19  

---

## 0. Purpose

Phase 3 turns the Phase 1/2 math + estimator code into a **trainable RL4CO-compatible algorithm** and a **reproducible training/evaluation harness** so Phase 4 experiments are straightforward.

Concretely, Phase 3 delivers:

- A Max@K policy-gradient implementation that plugs into RL4CO environments and policies.
- A POMO-based training loop integration (multi-start sampling per instance).
- Stability mechanisms (clipping, scaling/normalization options, logging).
- A Leader Reward baseline (for apples-to-apples comparison).
- Evaluation scripts to benchmark Max@K metrics at inference.

---

## 1. Key Definitions (mapping math ↔ RL4CO)

### 1.1 Notation

- `n`: number of sampled trajectories per instance during training.
  - In POMO-style training, **`n = num_starts`** (multi-start rollouts per instance).
- `k`: the `K` in Max@K objective (must satisfy `1 <= k <= n`).
- `reward`: RL4CO returns `reward` (often `-cost`), higher is better.
- `log_likelihood`: RL4CO policy output `log_likelihood` is $\log \pi_\theta(\tau)$ (sum over decode steps).

### 1.2 Estimator usage (Phase 2 outputs)

We already have:

- Reward estimator: `src.estimators.maxk_reward.maxk_reward_estimate`  
- Gradient score weights: `src.estimators.maxk_gradient.maxk_gradient_weights`  
- Variance reduction:
  - Sample-LOO: `src.estimators.baselines.apply_sample_loo` (requires `n > k`)
  - SubLOO: `src.estimators.baselines.subloo_weights` (requires `k >= 2`)

Critical constraint:

- In the loss we must use **stop-gradient semantics** through weights:
  - `loss = -(weights.detach() * log_likelihood).sum(dim=-1).mean()`

---

## 2. Integration Strategy (local first, upstream later)

### 2.1 Local development (this repo)

Implement RL4CO-compatible code **inside `principled-bestofk/code/src`** and import RL4CO as a dependency (`rl4co==0.6.0`).

Pros:
- Fast iteration, tests live next to the math.
- No fork maintenance while the research is changing quickly.

### 2.2 Upstreaming (optional deliverable after Phase 4 is stable)

Once training + evaluation are stable and results are compelling, port the minimal code to an RL4CO fork and open a PR:

- Add model class under RL4CO’s `rl4co.models.zoo`.
- Add configs/examples and tests aligned with RL4CO CI.
- Keep this repo as the “reference implementation + derivations”.

---

## 3. Prerequisites / Phase 2 exit criteria

Before starting Phase 3, ensure:

1. Phase 2 estimator tests pass locally:
   - `python3 -m pytest code/tests -q`
2. RL4CO is importable in the environment (`rl4co==0.6.0`).
3. We can instantiate an RL4CO env + policy and run a forward pass.
4. We have decided the default training mapping:
   - `n := num_starts` in POMO
   - initial default `k` (e.g., `k=8` with `n=16`)

---

## 4. Repo changes we expect in Phase 3

Phase 3 will likely add:

- `code/src/algorithms/`
  - `losses.py` (reusable objective implementations)
  - `maxk_pomo.py` (principled Max@K)
  - `leader_reward.py` (baseline)
- `code/src/experiments/`
  - `train_tsp.py` (first target task)
  - `train_cvrp.py` (optional in Phase 3; required in Phase 4)
  - `evaluate.py` (evaluation harness)
  - `diagnose_gradients.py` (gradient variance / ESS diagnostics)
- `code/tests/`
  - Lightweight “loss math” tests (no long training) for integration sanity.

---

## 5. Task-by-task execution plan

### 5.1 T3.1 — Create MaxK policy gradient class (RL4CO-compatible module)

**Goal:** Implement an RL4CO-compatible module that trains a policy using the **unbiased Max@K gradient weights** from Phase 2, while keeping the **objective reusable** across architectures.

#### 5.1.1 Design choice: where to plug in

Recommended approach: keep the *objective math* in a standalone loss helper, and subclass RL4CO’s POMO implementation to plug it into the training loop.

- Base class: `rl4co.models.zoo.pomo.POMO`
- Hook point: override `calculate_loss(...)` (called from `POMO.shared_step` during training)
- Objective: `src.algorithms.losses.MaxKLoss` (reusable across modules that expose `reward` + `log_likelihood`)

Why:
- Reuses RL4CO’s multistart decoding, reward computation, logging conventions.
- Keeps integration minimal and PR-friendly later.

#### 5.1.2 Proposed class + API

Create `MaxKPOMO` (name is flexible; keep explicit `MaxK` and `POMO`).

Core init parameters:

- `k: int` (Max@K parameter)
- `variance_reduction: str` in `{ "none", "sample_loo", "subloo" }`
- `stable_sort: bool = True` (deterministic tie handling)
- `start_node_mode: str` in `{ "pomo", "random" }`
  - `"pomo"`: use RL4CO default `env.select_start_nodes` (standard POMO)
  - `"random"`: pass `select_start_nodes_fn` to sample start nodes with replacement (closer to i.i.d.)
- (Optional) `weight_scale: None | "scale" | "norm" | int` (research knob; may bias objective)

#### 5.1.3 Loss implementation details

Inputs from `POMO.shared_step` during training:

- `reward`: shape `[batch, n]` where `n = num_starts`
- `log_likelihood`: shape `[batch, n]`

Compute:

1. `s = maxk_gradient_weights(reward, k)`
2. Apply variance reduction:
   - if `"sample_loo"`: `weights = apply_sample_loo(s, reward, k)` (require `n > k`)
   - if `"subloo"`: `weights = subloo_weights(reward, k)` (require `k >= 2`)
   - else: `weights = s`
3. Loss:
   - `loss = -(weights.detach() * log_likelihood).sum(dim=-1).mean()`

Note: the LOO baselines here are part of the *principled estimator* (variance reduction without bias), not just an ad-hoc “stability” tweak.

Recommended extra logging (for debugging/Phase 4):

- `rho_hat = maxk_reward_estimate(reward, k)` (per-instance Max@K reward estimate)
- `max_reward = reward.max(dim=-1).values`
- `mean_reward = reward.mean(dim=-1)`
- (Optional) `weights_mean`, `weights_std`, and NaN/inf checks

#### 5.1.4 Deliverables

- `code/src/algorithms/losses.py` (`MaxKLoss`)
- `code/src/algorithms/maxk_pomo.py`
- `code/src/algorithms/__init__.py` exports
- Minimal integration test(s) (fast):
  - Ensure shapes, constraints (`1 <= k <= n`, `n > k` for Sample-LOO, `k >= 2` for SubLOO)
  - Verify `.detach()` is used (no autograd path through sorting)
- Gradient sanity check (toy problem; fast):
  - Verify the surrogate produces an unbiased gradient on an enumeratable categorical toy Max@K objective.

#### 5.1.5 Done when

- A single training step runs without error on a small batch (`tsp`).
- Loss decreases in a short smoke run (e.g., 100–500 steps) without NaNs.
- Metrics/logging show reasonable magnitudes for `rho_hat` and `weights`.

---

### 5.2 T3.2 — Integrate with POMO training loop (training script)

**Goal:** Provide runnable scripts that train:

- Principled Max@K (T3.1)
- Baseline POMO (RL4CO built-in)
- Leader Reward (T3.4)

#### 5.2.1 Training entrypoints

Create scripts under `code/src/experiments/`:

- `train_tsp.py` (required first target)
- `train_cvrp.py` (optional in Phase 3; useful early if time allows)

Each script should support:

- selecting algorithm (`pomo`, `maxk_pomo`, `leader_reward`)
- choosing `n=num_starts`, `k`, and variance reduction mode
- basic Lightning trainer knobs (epochs, batch sizes, grad clipping, device)
- checkpoint output dir + seed control

#### 5.2.2 Recommended implementation pattern (simple Python, no Hydra)

In this repo, prefer a minimal, explicit training script:

1. Instantiate RL4CO env via `rl4co.envs.get_env(problem, generator_params=...)`.
2. Instantiate model:
   - POMO baseline: `rl4co.models.zoo.POMO(env, ...)`
   - MaxK: `src.algorithms.maxk_pomo.MaxKPOMO(env, k=..., ...)`
   - Leader: `src.algorithms.leader_reward.LeaderRewardPOMO(env, ...)`
3. Instantiate Lightning `Trainer` (or RL4COTrainer) with:
   - `gradient_clip_val`
   - checkpoint callback
   - logger (CSV/W&B if desired)
4. Run `trainer.fit(model)`.

Hydra-based configs can be added later (or at upstreaming time) if needed.

#### 5.2.3 Deliverables

- `code/src/experiments/train_tsp.py`
- (Optional) `code/src/experiments/train_cvrp.py`
- A small README section in the script docstring describing example commands.

#### 5.2.4 Done when

- You can run a short local training for all three algorithms and get checkpoints + logged metrics.
- A single config (k/n) is reproducible via seed.

---

### 5.3 T3.3 — Implement stability mechanisms (clipping, normalization)

**Goal:** Make training robust across `n` and `k` without hiding estimator bugs.

#### 5.3.1 Gradient clipping (must-have)

Use Lightning trainer gradient clipping:

- `gradient_clip_val` (default: `1.0`)
- `gradient_clip_algorithm="norm"`

This should be configured in training scripts (T3.2), not buried in the estimator.

#### 5.3.2 Weight/reward scaling (optional, clearly labeled)

Because this project emphasizes **principled (unbiased) gradients**, treat any scaling/normalization beyond LOO baselines as **experimental**:

- Safe default: no scaling (`None`)
- If enabled, implement as an explicit option (and document that it changes the effective objective / step scaling):
  - constant division by an integer factor (behaves like LR change)
  - running-std scaling (stability; may bias the objective)

#### 5.3.3 Numerical stability + debugging guards (must-have)

Add cheap checks in the algorithm loss path (enabled by a flag):

- verify `torch.isfinite(reward).all()`, `torch.isfinite(log_likelihood).all()`
- verify `torch.isfinite(weights).all()`
- optional: clamp extreme weights for debugging only (off by default)

#### 5.3.4 Done when

- Training no longer crashes for moderately large `n` (e.g., `n=64`) on CPU/GPU.
- NaN/inf detection provides actionable error messages.

---

### 5.4 T3.4 — Implement Leader Reward baseline (comparison baseline)

**Goal:** Implement the Wang et al. (2024) “Leader Reward” variant in the same RL4CO/POMO training loop for clean comparisons.

#### 5.4.1 Baseline definition (implementation-level)

Leader Reward is a heuristic modification to POMO advantages that boosts the best sample (“leader”) within the `n` trajectories for an instance.

Implementation requirements:

- Must share the same environment/policy and multistart decoding as POMO/MaxK.
- Must reduce to standard POMO when `alpha=0`.
- Must expose `alpha` (and any additional hyperparameters) explicitly.

#### 5.4.2 Deliverables

- `code/src/algorithms/leader_reward.py` implementing e.g. `LeaderRewardPOMO`
- Small unit test:
  - `alpha=0` matches baseline POMO loss numerically (up to floating error) for the same `reward` and `log_likelihood`.

#### 5.4.3 Done when

- Leader Reward runs end-to-end with the same training script interface as MaxK.
- We can sweep `alpha` without instability at small values.

---

### 5.5 T3.5 — Create evaluation harness (evaluation scripts)

**Goal:** Evaluate trained checkpoints on RL4CO datasets with **Max@K metrics**, and generate comparable outputs for Phase 4 plots/tables.

#### 5.5.1 Requirements

Evaluation should:

- Load a checkpoint for a given algorithm.
- Run inference with a chosen evaluation method:
  - `greedy`
  - `sampling` with `K_eval` samples and `select_best=True`
  - `multistart_greedy` (POMO-style inference)
  - `augment` variants if needed later
- Report:
  - average reward (or cost) over test instances
  - best-of-K reward for sampling-based evaluation
  - runtime / throughput
- Save structured results (e.g., `.pkl` or `.jsonl`) in a stable schema.
- (Research/Phase 4 alignment) Provide gradient-quality diagnostics:
  - weight concentration (ESS-style) for the per-sample loss weights/advantages
  - gradient variance proxy across repeated resampling (same instances, different RNG)

#### 5.5.2 Implementation approach

Because RL4CO’s CLI `rl4co/tasks/eval.py` only loads models defined in `rl4co.models.zoo`, the harness in this repo should:

1. Import RL4CO env + datasets.
2. Load our model class from `src.algorithms...` (for MaxK / Leader Reward), or RL4CO’s class (for baseline POMO).
3. Reuse RL4CO’s evaluation utilities (`rl4co.tasks.eval.evaluate_policy`) when convenient.

#### 5.5.3 Deliverables

- `code/src/experiments/evaluate.py`
  - CLI args: `--problem`, `--ckpt_path`, `--algorithm`, `--method`, `--k_eval`, `--num_instances`, `--seed`, `--device`, `--save_path`
- `code/src/experiments/diagnose_gradients.py`
  - CLI args: `--problem`, `--ckpt_path`, `--algorithm`, `--num_instances`, `--batch_size`, `--num_replicates`, `--seed`, `--device`, `--save_path`
- Documented example commands in the script docstring.

#### 5.5.4 Done when

- We can evaluate all algorithms on TSP50 test set and save results deterministically.
- The evaluation reports **best-of-K** when using sampling (matches the Phase 4 metric needs).

---

## 6. Suggested Week 5–6 schedule

### Week 5 (core integration)

1. T3.1 implement `MaxKLoss` + `MaxKPOMO` and basic unit + toy gradient tests.
2. T3.4 implement Leader Reward baseline and parity tests (early comparison baseline).
3. T3.2 create `train_tsp.py` and smoke-run training for all algorithms.
4. T3.3 add gradient clipping + NaN guards.

### Week 6 (baselines + harness)

5. T3.5 build evaluation + diagnostics scripts and produce first comparable numbers on TSP50.

---

## 7. Phase 3 exit criteria (ready for Phase 4)

Phase 3 is “done” when:

1. We can train MaxK and baselines end-to-end on TSP50.
2. We can evaluate checkpoints with a fixed evaluation protocol.
3. The pipeline is reproducible (seeded) and logs the key metrics needed for Phase 4:
   - reward curves
   - Max@K evaluation performance
   - (optional) gradient variance proxies
