# Task 3.1 — Implementation Notes (MaxK policy gradient class)

**Task:** T3.1 — Create MaxK policy gradient class (RL4CO-compatible module)  
**Status:** Implemented (local module + unit tests)  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- RL4CO-compatible algorithm module: `code/src/algorithms/maxk_pomo.py` (`MaxKPOMO`)
- Unit tests: `code/tests/test_maxk_pomo.py`

---

## 2. Design summary

### 2.1 Integration point

We subclass RL4CO's `rl4co.models.zoo.pomo.POMO` and override `calculate_loss(...)`.

- RL4CO calls `calculate_loss(td, batch, out, reward, log_likelihood)` from `POMO.shared_step(...)`.
- During training, RL4CO passes:
  - `reward`: `unbatchify(out["reward"], (n_aug, n_start))`, where `n_aug=0` in training, so the effective shape is `[batch, n_start]`.
  - `log_likelihood`: similarly unbatchified to `[batch, n_start]`.
- `MaxKPOMO.calculate_loss(...)` assumes these **unbatchified** 2D tensors; it raises `ValueError` if shapes are not `[batch, n]`.

### 2.2 Loss definition

We implement the principled Max@K REINFORCE-style loss:

```math
\mathcal{L}(\theta) = -\mathbb{E}\left[\sum_{i=1}^{n} w_i(R_{1:n}) \log \pi_\theta(\tau_i)\right],
```

with stop-gradient semantics through the weights:

- `weights` are computed under `torch.no_grad()`.
- The loss uses `weights.detach()`:
  - `loss = -(weights.detach() * log_likelihood).sum(dim=-1).mean()`

Weights come from Phase 2 estimators:

- Base Max@K score weights: `src.estimators.maxk_gradient.maxk_gradient_weights`
- Optional variance reduction:
  - Sample-LOO: `src.estimators.baselines.apply_sample_loo` (requires `n > k`)
  - SubLOO: `src.estimators.baselines.subloo_weights` (requires `k >= 2`)

We also compute `rho_hat = src.estimators.maxk_reward.maxk_reward_estimate(rewards, k)` under `no_grad` for logging/diagnostics.

### 2.3 Determinism and tie-handling

`stable_sort: bool = True` is threaded through to the estimator calls so that sorting tie-breaking is deterministic (stable w.r.t. original indices).

### 2.4 Hyperparameters / checkpoint hygiene

RL4CO's upstream constructors call `self.save_hyperparameters(...)` early. We override `save_hyperparameters` to always ignore `env` and `policy` to avoid Lightning warnings about storing `nn.Module` objects in the hparams dict.

---

## 3. Test coverage notes

`code/tests/test_maxk_pomo.py` validates:

- `calculate_loss(...)` matches a direct reference computation for:
  - `variance_reduction="none"`
  - `variance_reduction="sample_loo"`
  - `variance_reduction="subloo"`
- Sample-LOO constraint (`n > k`) raises a `ValueError`.
- Shape contract: rewards must be unbatchified `[batch, n]`.
- Optional smoke test (skipped if RL4CO envs are unavailable): instantiate a real RL4CO env and ensure the model constructs with reasonable `policy_kwargs`.

---

## 4. Known rough edges (environment)

Some RL4CO imports transitively import Matplotlib, which may try to write font caches and emit warnings in restricted environments. If needed, set `MPLCONFIGDIR` to a writable path (e.g., `/tmp/mplconfig`) when running tests.

