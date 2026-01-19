"""Gradient-quality diagnostics for trained checkpoints.

This script complements `evaluate.py` (inference metrics) by measuring training-time
signals that matter for *principled* estimators:

  - Gradient variance proxies across repeated resampling (same instances, different RNG)
  - Weight concentration diagnostics (ESS-style) for the per-sample loss weights

Supported algorithms:
  - `pomo`: RL4CO POMO baseline (shared-baseline REINFORCE)
  - `maxk_pomo`: principled Max@K loss (`src.algorithms.maxk_pomo.MaxKPOMO`)
  - `leader_reward`: heuristic baseline (`src.algorithms.leader_reward.LeaderRewardPOMO`)

Examples:
  - Diagnose gradient stability for a MaxK checkpoint on small TSP:
    `python3 code/src/experiments/diagnose_gradients.py --problem tsp --num_loc 20 --algorithm maxk_pomo --ckpt_path /path/to/last.ckpt --num_instances 128 --batch_size 64 --num_replicates 8 --save_path .tmp/diag/maxk_grad.json`
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
from torch.utils.data import DataLoader


_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_REPO_ROOT = _CODE_DIR.parent

Algorithm = Literal["pomo", "maxk_pomo", "leader_reward"]


@dataclass(frozen=True)
class DiagnosticsConfig:
    problem: str
    num_loc: int
    algorithm: Algorithm
    ckpt_path: Path
    num_instances: int
    batch_size: int
    num_replicates: int
    seed: int
    device: str
    save_path: Path


def _ensure_writable_caches(repo_root: Path) -> None:
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = tmp_dir / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    xdg_cache_dir = tmp_dir / "cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def parse_args(argv: list[str] | None = None) -> DiagnosticsConfig:
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
        "--num_instances",
        type=int,
        default=128,
        help="Number of instances used for diagnostics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for diagnostics.",
    )
    parser.add_argument(
        "--num_replicates",
        type=int,
        default=8,
        help="Number of repeated gradient computations (different RNG).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base RNG seed. Replicate i uses seed+ i.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (e.g., 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Output path for results (.jsonl, .json, or .pkl).",
    )

    args = parser.parse_args(argv)

    if args.num_loc < 2:
        raise ValueError(f"num_loc must be >= 2, got num_loc={args.num_loc}")
    if args.num_instances < 1:
        raise ValueError(
            f"num_instances must be >= 1, got num_instances={args.num_instances}"
        )
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got batch_size={args.batch_size}")
    if args.num_replicates < 1:
        raise ValueError(
            f"num_replicates must be >= 1, got num_replicates={args.num_replicates}"
        )

    return DiagnosticsConfig(
        problem=str(args.problem),
        num_loc=int(args.num_loc),
        algorithm=args.algorithm,
        ckpt_path=args.ckpt_path,
        num_instances=int(args.num_instances),
        batch_size=int(args.batch_size),
        num_replicates=int(args.num_replicates),
        seed=int(args.seed),
        device=str(args.device),
        save_path=args.save_path,
    )


def make_env(cfg: DiagnosticsConfig):
    from rl4co.envs import get_env

    return get_env(
        cfg.problem,
        generator_params={"num_loc": cfg.num_loc},
        device=cfg.device,
        seed=cfg.seed,
    )


def load_model(cfg: DiagnosticsConfig, env):
    if cfg.algorithm == "pomo":
        from rl4co.models.zoo.pomo import POMO as ModelCls
    elif cfg.algorithm == "maxk_pomo":
        from src.algorithms.maxk_pomo import MaxKPOMO as ModelCls
    elif cfg.algorithm == "leader_reward":
        from src.algorithms.leader_reward import LeaderRewardPOMO as ModelCls
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled algorithm: {cfg.algorithm}")

    device = torch.device(cfg.device)
    return ModelCls.load_from_checkpoint(cfg.ckpt_path, env=env, map_location=device)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().square().sum().item())
    return float(total**0.5)


def _grad_dot(
    parameters: list[torch.nn.Parameter],
    projections: list[torch.Tensor],
) -> float:
    dot = 0.0
    for param, proj in zip(parameters, projections):
        if param.grad is None:
            continue
        dot += float((param.grad.detach() * proj).sum().item())
    return dot


def _weights_for_algorithm(
    cfg: DiagnosticsConfig,
    model,
    td,
    batch,
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, weights) for the configured algorithm.

    All returned tensors are scalar loss and `[batch, n]` weights aligned with
    `log_likelihood`.
    """
    if cfg.algorithm == "maxk_pomo":
        loss_out = model.maxk_loss(
            reward, log_likelihood, scale_fn=model.advantage_scaler
        )
        return loss_out.loss, loss_out.weights

    baseline = reward.mean(dim=1, keepdim=True)
    advantage = reward - baseline

    if cfg.algorithm == "leader_reward":
        if model.alpha != 0.0:
            r_max, leader_idx = reward.max(dim=-1, keepdim=True)
            leader_mask = torch.zeros_like(reward)
            leader_mask.scatter_(-1, leader_idx, 1.0)
            beta = r_max - baseline
            advantage = advantage + (model.alpha * leader_mask * beta)

        advantage = model.advantage_scaler(advantage)
        if model.debug_clamp_weights is not None:
            advantage = advantage.clamp(
                -model.debug_clamp_weights, model.debug_clamp_weights
            )
        loss = -(advantage.detach() * log_likelihood).mean()
        return loss, advantage

    # cfg.algorithm == "pomo"
    advantage = model.advantage_scaler(advantage)
    loss = -(advantage.detach() * log_likelihood).mean()
    return loss, advantage


def diagnose(cfg: DiagnosticsConfig) -> dict[str, Any]:
    """Run gradient diagnostics and return a JSON-serializable dict."""
    if not cfg.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")

    from lightning.pytorch import seed_everything
    from rl4co.utils.ops import unbatchify

    seed_everything(cfg.seed, workers=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    model = load_model(cfg, env)
    model.to(cfg.device)
    model.train()
    model.policy.train()

    dataset = env.dataset(cfg.num_instances, phase="test")
    dataloader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, cfg.num_instances),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    policy_params = list(model.policy.parameters())
    if not policy_params:
        raise ValueError("Model policy has no parameters.")

    with torch.no_grad():
        torch.manual_seed(cfg.seed + 999)
        projections = [torch.randn_like(p) for p in policy_params]

    grad_norms: list[float] = []
    grad_dots: list[float] = []
    ess_means: list[float] = []
    weight_abs_max_means: list[float] = []

    t0 = time.time()
    for rep in range(cfg.num_replicates):
        seed_everything(cfg.seed + rep, workers=True)
        torch.manual_seed(cfg.seed + rep)
        np.random.seed(cfg.seed + rep)

        model.zero_grad(set_to_none=True)

        batch_losses: list[torch.Tensor] = []
        batch_ess: list[torch.Tensor] = []
        batch_abs_max: list[torch.Tensor] = []

        for batch in dataloader:
            batch = batch.to(cfg.device)
            td = env.reset(batch)

            n_start = model.num_starts
            if n_start is None:
                n_start = env.get_num_starts(td)

            out = model.policy(td, env, phase="train", num_starts=n_start)

            # During training RL4CO uses n_aug=0 (no augmentation) and unbatchifies
            # to [batch, n_start]. We mirror that here.
            reward = unbatchify(out["reward"], (0, n_start))
            log_likelihood = unbatchify(out["log_likelihood"], (0, n_start))

            loss, weights = _weights_for_algorithm(
                cfg, model, td, batch, reward, log_likelihood
            )
            batch_losses.append(loss)

            from src.algorithms.losses import effective_sample_size

            batch_ess.append(effective_sample_size(weights))
            batch_abs_max.append(weights.abs().max(dim=-1).values)

        loss_rep = torch.stack(batch_losses).mean()
        loss_rep.backward()

        grad_norms.append(_grad_norm(policy_params))
        grad_dots.append(_grad_dot(policy_params, projections))
        ess_means.append(float(torch.cat(batch_ess).mean().item()))
        weight_abs_max_means.append(float(torch.cat(batch_abs_max).mean().item()))

    wall_time = time.time() - t0

    grad_norms_arr = np.asarray(grad_norms, dtype=np.float64)
    grad_dots_arr = np.asarray(grad_dots, dtype=np.float64)
    ess_arr = np.asarray(ess_means, dtype=np.float64)
    abs_max_arr = np.asarray(weight_abs_max_means, dtype=np.float64)

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "config": asdict(cfg)
        | {"ckpt_path": str(cfg.ckpt_path), "save_path": str(cfg.save_path)},
        "metrics": {
            "grad_norm_mean": float(grad_norms_arr.mean()),
            "grad_norm_std": float(grad_norms_arr.std(ddof=0)),
            "grad_proj_mean": float(grad_dots_arr.mean()),
            "grad_proj_std": float(grad_dots_arr.std(ddof=0)),
            "ess_mean": float(ess_arr.mean()),
            "ess_std": float(ess_arr.std(ddof=0)),
            "weights_abs_max_mean": float(abs_max_arr.mean()),
            "weights_abs_max_std": float(abs_max_arr.std(ddof=0)),
            "wall_time_sec": float(wall_time),
        },
        "per_replicate": {
            "grad_norm": grad_norms,
            "grad_proj": grad_dots,
            "ess_mean": ess_means,
            "weights_abs_max_mean": weight_abs_max_means,
        },
    }


def save_result(result: dict[str, Any], save_path: Path) -> None:
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
    result = diagnose(cfg)
    save_result(result, cfg.save_path)

    metrics = result["metrics"]
    print(
        "Diagnostics complete:",
        f"grad_norm={metrics['grad_norm_mean']:.4g}±{metrics['grad_norm_std']:.2g}",
        f"ESS={metrics['ess_mean']:.3g}±{metrics['ess_std']:.2g}",
        f"saved={cfg.save_path}",
    )


if __name__ == "__main__":
    main()
