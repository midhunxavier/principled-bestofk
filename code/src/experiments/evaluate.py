"""Evaluate trained checkpoints on RL4CO datasets.

This script implements Phase 3 Task 3.5 (T3.5): create an evaluation harness
that loads checkpoints and reports Max@K / best-of-K metrics.

Supported algorithms:
  - `pomo`: RL4CO built-in POMO baseline
  - `maxk_pomo`: principled Max@K loss (`src.algorithms.maxk_pomo.MaxKPOMO`)
  - `leader_reward`: Leader Reward baseline (`src.algorithms.leader_reward.LeaderRewardPOMO`)

Supported evaluation methods (via `rl4co.tasks.eval.evaluate_policy`):
  - `greedy`
  - `sampling` (best-of-K via `--k_eval`)
  - `multistart_greedy`
  - `augment`, `augment_dihedral_8`, and multistart+augment variants

Examples:
  - Evaluate a POMO checkpoint on TSP50 with greedy decoding:
    `python3 code/src/experiments/evaluate.py --problem tsp --num_loc 50 --algorithm pomo --ckpt_path /path/to/last.ckpt --method greedy --num_instances 1000 --save_path results.jsonl`

  - Best-of-128 sampling evaluation (Max@128 metric):
    `python3 code/src/experiments/evaluate.py --problem tsp --num_loc 50 --algorithm maxk_pomo --ckpt_path /path/to/last.ckpt --method sampling --k_eval 128 --num_instances 1000 --save_path results.jsonl`

Note:
  If you prefer running as a module, use:
    `PYTHONPATH=code python3 -m src.experiments.evaluate ...`
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_REPO_ROOT = _CODE_DIR.parent

Algorithm = Literal["pomo", "maxk_pomo", "leader_reward"]
EvalMethod = Literal[
    "greedy",
    "sampling",
    "multistart_greedy",
    "augment",
    "augment_dihedral_8",
    "multistart_greedy_augment",
    "multistart_greedy_augment_dihedral_8",
]


@dataclass(frozen=True)
class EvalConfig:
    problem: str
    num_loc: int
    algorithm: Algorithm
    ckpt_path: Path
    method: EvalMethod
    k_eval: int
    num_instances: int
    seed: int
    device: str
    batch_size: int
    save_path: Path
    save_actions: bool


def _ensure_writable_caches(repo_root: Path) -> None:
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = tmp_dir / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    xdg_cache_dir = tmp_dir / "cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--problem",
        type=str,
        default="tsp",
        help="RL4CO environment name (e.g., 'tsp', 'cvrp').",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Graph size for routing problems (e.g., TSP50 => 50).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=("pomo", "maxk_pomo", "leader_reward"),
        default="maxk_pomo",
        help="Which algorithm class to use when loading the checkpoint.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Path to a Lightning checkpoint (*.ckpt).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "greedy",
            "sampling",
            "multistart_greedy",
            "augment",
            "augment_dihedral_8",
            "multistart_greedy_augment",
            "multistart_greedy_augment_dihedral_8",
        ),
        default="greedy",
        help="Evaluation method (RL4CO decoding / inference protocol).",
    )
    parser.add_argument(
        "--k_eval",
        type=int,
        default=128,
        help="Number of samples for `--method sampling` (best-of-K / Max@K metric).",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1_000,
        help="Number of test instances to evaluate on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed used for dataset generation and sampling evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (e.g., 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Output path for results (.jsonl, .json, or .pkl).",
    )
    parser.add_argument(
        "--save_actions",
        action="store_true",
        help="Save padded action sequences (can be large).",
    )

    args = parser.parse_args(argv)

    if args.num_loc < 2:
        raise ValueError(f"num_loc must be >= 2, got num_loc={args.num_loc}")
    if args.k_eval < 1:
        raise ValueError(f"k_eval must be >= 1, got k_eval={args.k_eval}")
    if args.num_instances < 1:
        raise ValueError(
            f"num_instances must be >= 1, got num_instances={args.num_instances}"
        )
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got batch_size={args.batch_size}")

    return EvalConfig(
        problem=str(args.problem),
        num_loc=int(args.num_loc),
        algorithm=args.algorithm,
        ckpt_path=args.ckpt_path,
        method=args.method,
        k_eval=int(args.k_eval),
        num_instances=int(args.num_instances),
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch_size),
        save_path=args.save_path,
        save_actions=bool(args.save_actions),
    )


def make_env(cfg: EvalConfig):
    """Create an RL4CO environment configured for evaluation."""

    from rl4co.envs import get_env

    generator_params: dict[str, Any] = {"num_loc": cfg.num_loc}
    return get_env(
        cfg.problem,
        generator_params=generator_params,
        device=cfg.device,
        seed=cfg.seed,
    )


def load_model(cfg: EvalConfig, env):
    """Load a Lightning module checkpoint and return the instantiated model."""

    if cfg.algorithm == "pomo":
        from rl4co.models.zoo.pomo import POMO as ModelCls

        load_kwargs: dict[str, Any] = {}
    elif cfg.algorithm == "maxk_pomo":
        from src.algorithms.maxk_pomo import MaxKPOMO as ModelCls

        load_kwargs = {}
    elif cfg.algorithm == "leader_reward":
        from src.algorithms.leader_reward import LeaderRewardPOMO as ModelCls

        load_kwargs = {}
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled algorithm: {cfg.algorithm}")

    device = torch.device(cfg.device)
    return ModelCls.load_from_checkpoint(
        cfg.ckpt_path,
        env=env,
        map_location=device,
        **load_kwargs,
    )


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    """Run evaluation and return a JSON-serializable result dict.

    Args:
        cfg: Evaluation configuration.

    Returns:
        Dict containing configuration metadata, aggregate metrics, and per-instance rewards.

    Raises:
        FileNotFoundError: If `cfg.ckpt_path` does not exist.
        ValueError: If evaluation outputs are malformed.
    """

    if not cfg.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")

    from lightning.pytorch import seed_everything
    from rl4co.tasks.eval import evaluate_policy

    seed_everything(cfg.seed, workers=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    model = load_model(cfg, env)
    model.to(cfg.device)
    model.eval()

    policy = model.policy
    policy.eval()

    dataset = env.dataset(cfg.num_instances, phase="test")

    t0 = time.time()
    retvals = evaluate_policy(
        env,
        policy,
        dataset,
        method=cfg.method,
        batch_size=min(cfg.batch_size, cfg.num_instances),
        auto_batch_size=False,
        samples=cfg.k_eval,
        progress=False,
    )
    wall_time = time.time() - t0

    rewards = retvals.get("rewards", None)
    if rewards is None or not isinstance(rewards, torch.Tensor):
        raise ValueError("evaluate_policy did not return a tensor under key 'rewards'.")

    rewards_cpu = rewards.detach().cpu().flatten()
    avg_reward = float(rewards_cpu.mean().item())
    reward_std = float(rewards_cpu.std(unbiased=False).item())

    inference_time = float(retvals.get("inference_time", wall_time))
    throughput = (
        float(cfg.num_instances) / inference_time if inference_time > 0 else 0.0
    )

    actions_payload: list[list[int]] | None = None
    if cfg.save_actions:
        actions = retvals.get("actions", None)
        if actions is None or not isinstance(actions, torch.Tensor):
            raise ValueError(
                "save_actions requested, but evaluate_policy did not return 'actions'."
            )
        actions_payload = actions.detach().cpu().to(torch.int64).tolist()

    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "config": asdict(cfg)
        | {"ckpt_path": str(cfg.ckpt_path), "save_path": str(cfg.save_path)},
        "metrics": {
            "avg_reward": avg_reward,
            "avg_cost": -avg_reward,
            "reward_std": reward_std,
            "inference_time_sec": inference_time,
            "wall_time_sec": wall_time,
            "throughput_instances_per_sec": throughput,
            "method": cfg.method,
            "k_eval": cfg.k_eval if cfg.method == "sampling" else None,
        },
        "rewards": rewards_cpu.to(torch.float32).tolist(),
        "actions": actions_payload,
    }

    # Include any extra scalars RL4CO provides (e.g., avg_reward tensor).
    for key in ("avg_reward",):
        if key in retvals:
            result["metrics"][f"rl4co_{key}"] = _as_float(retvals[key])

    return result


def save_result(result: dict[str, Any], save_path: Path) -> None:
    """Save evaluation results to disk.

    Args:
        result: Result dict from :func:`evaluate`.
        save_path: Destination file path. Supported extensions: .jsonl, .json, .pkl.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()

    if suffix == ".jsonl":
        with save_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        return

    if suffix == ".json":
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        return

    if suffix == ".pkl":
        with save_path.open("wb") as f:
            pickle.dump(result, f)
        return

    raise ValueError(
        f"Unsupported save_path extension {save_path.suffix!r}; use .jsonl, .json, or .pkl"
    )


def main(argv: list[str] | None = None) -> None:
    _ensure_writable_caches(_REPO_ROOT)

    cfg = parse_args(argv)
    result = evaluate(cfg)
    save_result(result, cfg.save_path)

    metrics = result["metrics"]
    avg_reward = metrics["avg_reward"]
    avg_cost = metrics["avg_cost"]
    throughput = metrics["throughput_instances_per_sec"]
    print(
        "Eval complete:",
        f"avg_reward={avg_reward:.6f}",
        f"avg_cost={avg_cost:.6f}",
        f"time={metrics['inference_time_sec']:.3f}s",
        f"throughput={throughput:.2f} inst/s",
        f"saved={cfg.save_path}",
    )


if __name__ == "__main__":
    main()
