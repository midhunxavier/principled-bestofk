# Task 3.4 — Implementation Notes (Leader Reward baseline)

**Task:** T3.4 — Implement Leader Reward baseline (comparison baseline)  
**Status:** Implemented (local module + parity test)  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- RL4CO-compatible baseline module: `code/src/algorithms/leader_reward.py` (`LeaderRewardPOMO`)
- Unit tests: `code/tests/test_leader_reward_pomo.py`
- Training script integration: `code/src/experiments/train_tsp.py` (`--algorithm leader_reward`, `--alpha`)

---

## 2. Baseline definition

Leader Reward modifies POMO’s shared-baseline advantage by adding an extra bonus to the best (“leader”) sample:

```math
A_i = (R_i - \mathrm{mean}(R)) + \alpha \cdot \mathbf{1}[i = \arg\max_j R_j] \cdot \beta
```

with the implemented bonus term:

```math
\beta = R_{\max} - \mathrm{mean}(R)
```

The implementation guarantees:

- Setting `alpha=0` reduces to standard POMO loss.
- The environment/policy/multi-start decoding path is identical to RL4CO POMO (subclass of `rl4co.models.zoo.pomo.POMO` overriding `calculate_loss(...)` only).

---

## 3. Test coverage notes

`code/tests/test_leader_reward_pomo.py` validates:

- `alpha=0` matches RL4CO `POMO.calculate_loss(...)` numerically for the same `reward` and `log_likelihood`.
- A fixed reference computation matches `LeaderRewardPOMO.calculate_loss(...)` for `alpha>0`.
- Shape contract enforcement (expects unbatchified `[batch, n]`).

