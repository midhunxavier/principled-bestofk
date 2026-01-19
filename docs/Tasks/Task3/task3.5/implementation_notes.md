# Task 3.5 — Implementation Notes (Evaluation harness)

**Task:** T3.5 — Create evaluation harness (evaluation scripts)  
**Status:** Implemented (script + smoke tests)  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- Evaluation script: `code/src/experiments/evaluate.py`
  - Loads a checkpoint for `pomo`, `maxk_pomo`, or `leader_reward`
  - Runs RL4CO evaluation with Max@K / best-of-K support (`--method sampling --k_eval K`)
  - Saves structured results (`.jsonl`, `.json`, or `.pkl`)
- Gradient diagnostics script: `code/src/experiments/diagnose_gradients.py`
  - Computes gradient variance proxies across repeated resampling
  - Computes ESS-style weight concentration diagnostics
  - Saves structured results (`.jsonl`, `.json`, or `.pkl`)
- Unit tests: `code/tests/test_evaluate_script.py`
  - End-to-end smoke: create a tiny `MaxKPOMO` checkpoint, evaluate on a small TSP dataset, and validate output schema
- Unit tests: `code/tests/test_diagnose_gradients_script.py`
  - End-to-end smoke: create a tiny `MaxKPOMO` checkpoint, run diagnostics with a few replicates, and validate output schema

---

## 2. Implementation overview

### 2.1 Checkpoint loading

The harness uses Lightning’s `load_from_checkpoint` with an environment passed explicitly:

- Baseline: `rl4co.models.zoo.pomo.POMO.load_from_checkpoint(..., env=env)`
- Principled: `src.algorithms.maxk_pomo.MaxKPOMO.load_from_checkpoint(..., env=env)`
- Leader Reward: `src.algorithms.leader_reward.LeaderRewardPOMO.load_from_checkpoint(..., env=env)`

At evaluation time we always call `evaluate_policy(env, model.policy, dataset, ...)` so the decoding behavior is driven by the policy.

### 2.2 Dataset and determinism

The script evaluates on `dataset = env.dataset(num_instances, phase="test")`.

By default, RL4CO environments in this repo are constructed without a `test_file`, so `env.dataset(...)` generates instances from the environment generator rather than loading from disk.

Determinism:

- The script seeds `lightning.pytorch.seed_everything`, `torch.manual_seed`, and `numpy.random.seed`.
- The environment is created with `seed=<cfg.seed>`, which sets RL4CO’s internal RNG.

This makes the generated evaluation dataset and sampling reproducible for a fixed `(seed, device)` (subject to usual GPU nondeterminism caveats).

### 2.3 Evaluation methods and Max@K metric

Evaluation is delegated to RL4CO’s `rl4co.tasks.eval.evaluate_policy`.

Supported methods include:

- `greedy`
- `sampling` (best-of-K via `--k_eval`)
- `multistart_greedy`
- `augment`, `augment_dihedral_8`, and multistart+augment variants

For Max@K / best-of-K:

- Use `--method sampling --k_eval K`
- RL4CO’s sampling evaluator uses `select_best=True` by default, so the returned per-instance reward corresponds to best-of-`K`.

---

### 2.4 Gradient diagnostics (variance + ESS)

`diagnose_gradients.py` measures estimator quality signals that complement inference metrics:

- **Gradient variance proxy:** repeats gradient computation on the *same instances* across multiple RNG seeds (replicates), and reports variance across the resulting gradients via a norm and a fixed random projection.
- **ESS-style weight concentration:** computes a weight concentration metric from the per-sample loss weights/advantages:

```math
\\mathrm{ESS}(w) = \\frac{(\\sum_i w_i)^2}{\\sum_i w_i^2}
```

This is used as a diagnostic for weight concentration / cancellation (it is not a strict importance-sampling ESS when weights are signed).

---

## 3. Output schema

Results are written as a single record with `schema_version=1` containing:

- `config`: the full CLI config (with `ckpt_path` and `save_path` serialized as strings)
- `metrics`:
  - `avg_reward`: mean reward over instances
  - `avg_cost`: `-avg_reward` (for routing tasks where reward is typically `-cost`)
  - `reward_std`
  - `inference_time_sec`, `wall_time_sec`, `throughput_instances_per_sec`
  - `method`, and `k_eval` (only set when `method == "sampling"`)
- `rewards`: per-instance rewards as a list
- `actions`: optional padded actions list when `--save_actions` is set (can be large)

---

## 4. Usage

Examples:

- Greedy evaluation (TSP50):
  - `python3 code/src/experiments/evaluate.py --problem tsp --num_loc 50 --algorithm pomo --ckpt_path /path/to/last.ckpt --method greedy --num_instances 1000 --seed 1234 --device cpu --batch_size 256 --save_path .tmp/eval/tsp50_greedy.jsonl`
- Best-of-128 sampling (Max@128 metric):
  - `python3 code/src/experiments/evaluate.py --problem tsp --num_loc 50 --algorithm maxk_pomo --ckpt_path /path/to/last.ckpt --method sampling --k_eval 128 --num_instances 1000 --seed 1234 --device cpu --batch_size 256 --save_path .tmp/eval/tsp50_sampling_k128.jsonl`
- POMO-style multistart greedy (uses `num_starts=num_loc` in RL4CO’s default evaluator):
  - `python3 code/src/experiments/evaluate.py --problem tsp --num_loc 50 --algorithm pomo --ckpt_path /path/to/last.ckpt --method multistart_greedy --num_instances 1000 --seed 1234 --device cpu --batch_size 256 --save_path .tmp/eval/tsp50_multistart.jsonl`

Gradient diagnostics:

- Diagnostics for gradient stability + ESS (small run):
  - `python3 code/src/experiments/diagnose_gradients.py --problem tsp --num_loc 20 --algorithm maxk_pomo --ckpt_path /path/to/last.ckpt --num_instances 128 --batch_size 64 --num_replicates 8 --seed 1234 --device cpu --save_path .tmp/diag/tsp20_maxk_grad.json`
