"""Train POMO-based models on TSP with RL4CO.

This script implements Phase 3 Task 3.2 (T3.2): integrate our algorithms with
RL4CO's POMO training loop (multi-start rollouts).

Supported algorithms:
  - `pomo`: RL4CO built-in POMO baseline
  - `maxk_pomo`: principled Max@K loss (`src.algorithms.maxk_pomo.MaxKPOMO`)
  - `leader_reward`: Leader Reward baseline (`src.algorithms.leader_reward.LeaderRewardPOMO`)

Examples:
  - Baseline POMO (small smoke run):
    `python3 code/src/experiments/train_tsp.py --algorithm pomo --num_loc 20 --max_epochs 1`

  - Principled Max@K (k=4) with SubLOO:
    `python3 code/src/experiments/train_tsp.py --algorithm maxk_pomo --num_loc 20 --k 4 --variance_reduction subloo --max_epochs 1`

  - Leader Reward baseline:
    `python3 code/src/experiments/train_tsp.py --algorithm leader_reward --num_loc 20 --alpha 0.5 --max_epochs 1`

Note:
  If you prefer running as a module, use:
    `PYTHONPATH=code python3 -m src.experiments.train_tsp ...`
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_REPO_ROOT = _CODE_DIR.parent

Algorithm = Literal["pomo", "maxk_pomo", "leader_reward"]
VarianceReduction = Literal["none", "sample_loo", "subloo", "hybrid"]
WeightNormalization = Literal["none", "zscore", "sum_to_zero"]
RewardScale = int | Literal["scale", "norm"] | None


@dataclass(frozen=True)
class TrainConfig:
    algorithm: Algorithm
    seed: int
    num_loc: int
    num_starts: int | None
    num_augment: int
    output_dir: Path
    run_name: str
    data_dir: Path
    generate_default_data: bool
    max_epochs: int
    batch_size: int
    train_data_size: int
    val_data_size: int
    test_data_size: int
    lr: float
    accelerator: str
    devices: str | int
    precision: str | int
    gradient_clip_val: float | None
    log_every_n_steps: int
    enable_progress_bar: bool
    limit_train_batches: float | int | None
    limit_val_batches: float | int | None
    k: int
    variance_reduction: VarianceReduction
    weight_normalization: WeightNormalization
    min_gap_scale: float
    hybrid_lambda: float
    stable_sort: bool
    alpha: float
    reward_scale: RewardScale
    check_numerics: bool
    debug_clamp_weights: float | None
    policy_kwargs: dict[str, Any]


def _ensure_writable_caches(repo_root: Path) -> None:
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = tmp_dir / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    xdg_cache_dir = tmp_dir / "cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _parse_devices(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _parse_precision(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_limit(value: str) -> float | int:
    try:
        return int(value)
    except ValueError:
        return float(value)


def _parse_reward_scale(value: str) -> RewardScale:
    lowered = value.strip().lower()
    if lowered in ("none", "null", ""):
        return None
    if lowered == "scale":
        return "scale"
    if lowered == "norm":
        return "norm"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"reward_scale must be one of {{none, scale, norm}} or an int, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"reward_scale int must be > 0, got {parsed} from {value!r}"
        )
    return parsed


def _default_run_name(
    algorithm: Algorithm, num_loc: int, num_starts: int | None, seed: int
) -> str:
    n_str = "auto" if num_starts is None else str(num_starts)
    return f"{algorithm}_tsp{num_loc}_n{n_str}_seed{seed}"


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=("pomo", "maxk_pomo", "leader_reward"),
        default="maxk_pomo",
        help="Which algorithm to train.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed.")

    parser.add_argument(
        "--num_loc", type=int, default=50, help="TSP size (number of nodes)."
    )
    parser.add_argument(
        "--num_starts",
        type=int,
        default=None,
        help="Number of multi-start rollouts per instance (n). If omitted, uses the env default.",
    )
    parser.add_argument(
        "--num_augment",
        type=int,
        default=1,
        help="Number of augmentations for val/test (training uses 0 regardless).",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO_ROOT / ".tmp" / "runs" / "tsp",
        help="Root directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Run name (subdirectory under output_dir). If omitted, a deterministic name is used.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_REPO_ROOT / ".tmp" / "rl4co_data",
        help="Directory RL4CO uses for dataset files (if any).",
    )
    parser.add_argument(
        "--generate_default_data",
        action="store_true",
        help="Generate RL4CO default datasets in data_dir if they don't exist.",
    )

    parser.add_argument(
        "--max_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--train_data_size",
        type=int,
        default=10_000,
        help="Number of training instances (dataset size).",
    )
    parser.add_argument(
        "--val_data_size",
        type=int,
        default=1_000,
        help="Number of validation instances (dataset size).",
    )
    parser.add_argument(
        "--test_data_size",
        type=int,
        default=1_000,
        help="Number of test instances (dataset size).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")

    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Lightning accelerator ('cpu', 'gpu', 'mps', or 'auto').",
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default="auto",
        help="Lightning devices (int, 'auto', or device specification).",
    )
    parser.add_argument(
        "--precision",
        type=_parse_precision,
        default="32-true",
        help="Lightning precision (e.g., '32-true', '16-mixed', or an int).",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clip value (set to 0 to disable).",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Logging frequency (in optimizer steps).",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable Lightning progress bar.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=_parse_limit,
        default=None,
        help="Optionally limit train batches (float fraction or int).",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=_parse_limit,
        default=None,
        help="Optionally limit val batches (float fraction or int).",
    )

    parser.add_argument(
        "--k", type=int, default=8, help="Max@K parameter (for maxk_pomo)."
    )
    parser.add_argument(
        "--variance_reduction",
        type=str,
        choices=("none", "sample_loo", "subloo", "hybrid"),
        default="none",
        help="Variance reduction mode for maxk_pomo.",
    )
    parser.add_argument(
        "--weight_normalization",
        type=str,
        choices=("none", "zscore", "sum_to_zero"),
        default="zscore",
        help="Weight normalization mode for maxk_pomo (default: zscore).",
    )
    parser.add_argument(
        "--min_gap_scale",
        type=float,
        default=0.01,
        help="Minimum gap scale for SubLOO to prevent zero gradients (default: 0.01).",
    )
    parser.add_argument(
        "--hybrid_lambda",
        type=float,
        default=0.5,
        help="Blending coefficient for hybrid mode: 1.0=SubLOO, 0.0=POMO (default: 0.5).",
    )
    parser.add_argument(
        "--unstable_sort",
        action="store_true",
        help="Use unstable sorting (non-deterministic tie-breaking) in Max@K estimators.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Leader Reward bonus scaling (for leader_reward).",
    )

    parser.add_argument(
        "--reward_scale",
        type=_parse_reward_scale,
        default=None,
        help=(
            "Experimental scaling/normalization applied to advantages/weights. "
            "Use 'norm', 'scale', or an integer divisor."
        ),
    )
    parser.add_argument(
        "--check_numerics",
        action="store_true",
        help="Enable NaN/inf guards in custom loss implementations.",
    )
    parser.add_argument(
        "--debug_clamp_weights",
        type=float,
        default=None,
        help=(
            "Debug-only clamp of loss weights/advantages (biases the gradient; off by default)."
        ),
    )

    parser.add_argument(
        "--embed_dim", type=int, default=None, help="Policy embedding dimension."
    )
    parser.add_argument(
        "--num_heads", type=int, default=None, help="Policy attention heads."
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=None,
        help="Policy encoder layers.",
    )

    args = parser.parse_args(argv)

    algorithm: Algorithm = args.algorithm
    variance_reduction: VarianceReduction = args.variance_reduction
    stable_sort = not args.unstable_sort

    policy_kwargs: dict[str, Any] = {}
    if args.embed_dim is not None:
        policy_kwargs["embed_dim"] = args.embed_dim
    if args.num_heads is not None:
        policy_kwargs["num_heads"] = args.num_heads
    if args.num_encoder_layers is not None:
        policy_kwargs["num_encoder_layers"] = args.num_encoder_layers

    run_name = args.run_name or _default_run_name(
        algorithm, args.num_loc, args.num_starts, args.seed
    )

    if args.gradient_clip_val == 0:
        gradient_clip_val = None
    else:
        gradient_clip_val = float(args.gradient_clip_val)

    if algorithm == "maxk_pomo":
        n = args.num_loc if args.num_starts is None else args.num_starts
        if args.k < 1 or args.k > n:
            raise ValueError(f"k must satisfy 1 <= k <= n, got k={args.k}, n={n}")
        if variance_reduction == "sample_loo" and n <= args.k:
            raise ValueError(
                f"sample_loo requires n > k, got n={n}, k={args.k} "
                "(set --variance_reduction none/subloo or increase --num_starts)"
            )
        if variance_reduction == "subloo" and args.k < 2:
            raise ValueError("subloo requires k >= 2")

    if algorithm == "leader_reward" and args.alpha < 0:
        raise ValueError(f"alpha must be >= 0, got alpha={args.alpha}")

    if args.debug_clamp_weights is not None and args.debug_clamp_weights <= 0:
        raise ValueError(
            "debug_clamp_weights must be > 0 when provided, got "
            f"debug_clamp_weights={args.debug_clamp_weights}"
        )

    return TrainConfig(
        algorithm=algorithm,
        seed=args.seed,
        num_loc=args.num_loc,
        num_starts=args.num_starts,
        num_augment=args.num_augment,
        output_dir=args.output_dir,
        run_name=run_name,
        data_dir=args.data_dir,
        generate_default_data=bool(args.generate_default_data),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        lr=args.lr,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=not args.no_progress_bar,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        k=args.k,
        variance_reduction=variance_reduction,
        weight_normalization=args.weight_normalization,
        min_gap_scale=float(args.min_gap_scale),
        hybrid_lambda=float(args.hybrid_lambda),
        stable_sort=stable_sort,
        alpha=args.alpha,
        reward_scale=args.reward_scale,
        check_numerics=bool(args.check_numerics),
        debug_clamp_weights=args.debug_clamp_weights,
        policy_kwargs=policy_kwargs,
    )


def make_env(cfg: TrainConfig):
    from rl4co.envs import get_env

    return get_env(
        "tsp",
        generator_params={"num_loc": cfg.num_loc},
        data_dir=str(cfg.data_dir),
    )


def make_model(cfg: TrainConfig, env):
    from rl4co.models.zoo.pomo import POMO

    if cfg.algorithm == "pomo":
        model_cls = POMO
        model_kwargs: dict[str, Any] = {"reward_scale": cfg.reward_scale}
    elif cfg.algorithm == "maxk_pomo":
        from src.algorithms.maxk_pomo import MaxKPOMO

        model_cls = MaxKPOMO
        model_kwargs = {
            "k": cfg.k,
            "variance_reduction": cfg.variance_reduction,
            "weight_normalization": cfg.weight_normalization,
            "min_gap_scale": cfg.min_gap_scale,
            "hybrid_lambda": cfg.hybrid_lambda,
            "stable_sort": cfg.stable_sort,
            "reward_scale": cfg.reward_scale,
            "check_numerics": cfg.check_numerics,
            "debug_clamp_weights": cfg.debug_clamp_weights,
        }
    elif cfg.algorithm == "leader_reward":
        from src.algorithms.leader_reward import LeaderRewardPOMO

        model_cls = LeaderRewardPOMO
        model_kwargs = {
            "alpha": cfg.alpha,
            "reward_scale": cfg.reward_scale,
            "check_numerics": cfg.check_numerics,
            "debug_clamp_weights": cfg.debug_clamp_weights,
        }
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled algorithm: {cfg.algorithm}")

    return model_cls(
        env,
        num_augment=cfg.num_augment,
        num_starts=cfg.num_starts,
        policy_kwargs=cfg.policy_kwargs,
        batch_size=cfg.batch_size,
        val_batch_size=cfg.batch_size,
        test_batch_size=cfg.batch_size,
        train_data_size=cfg.train_data_size,
        val_data_size=cfg.val_data_size,
        test_data_size=cfg.test_data_size,
        data_dir=str(cfg.data_dir),
        generate_default_data=cfg.generate_default_data,
        optimizer_kwargs={"lr": cfg.lr},
        **model_kwargs,
    )


def make_trainer(cfg: TrainConfig, run_dir: Path):
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from rl4co.utils.trainer import RL4COTrainer

    logger = CSVLogger(save_dir=str(cfg.output_dir), name=cfg.run_name)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch:03d}",
        monitor="val/reward",
        mode="max",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
    ]

    return RL4COTrainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm="norm",
        precision=cfg.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.enable_progress_bar,
        deterministic=True,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )


def main(argv: list[str] | None = None) -> None:
    _ensure_writable_caches(_REPO_ROOT)

    cfg = parse_args(argv)

    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    from lightning.pytorch import seed_everything

    seed_everything(cfg.seed, workers=True)

    env = make_env(cfg)
    model = make_model(cfg, env)
    trainer = make_trainer(cfg, run_dir)

    trainer.fit(model)


if __name__ == "__main__":
    main()
