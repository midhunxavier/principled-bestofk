"""Phase 4 (T4.1) TSP experiments runner + summarizer.

This script is the implementation vehicle for PRD Task 4.1:

    | T4.1 | TSP experiments (50, 100 nodes) | Results tables, training curves |

It is designed to be run in Google Colab (TPU or GPU) and produces:

1) Training runs via `src.experiments.train_tsp`
2) Evaluation JSON artifacts via `src.experiments.evaluate`
3) A markdown/CSV results table + PNG training curves

Usage (typical):

  - Train TSP50 + TSP100 for all algorithms:
    `PYTHONPATH=code python3 -m src.experiments.phase4_tsp_experiments train --runs_dir /path/runs --artifacts_dir /path/artifacts --accelerator tpu --devices 8`

  - Evaluate all trained checkpoints (greedy + best-of-K sampling):
    `PYTHONPATH=code python3 -m src.experiments.phase4_tsp_experiments eval --runs_dir /path/runs --artifacts_dir /path/artifacts --k_eval 128`

  - Summarize into tables + curves:
    `PYTHONPATH=code python3 -m src.experiments.phase4_tsp_experiments summarize --runs_dir /path/runs --artifacts_dir /path/artifacts`
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


Algorithm = Literal["pomo", "leader_reward", "maxk_pomo"]
EvalMethod = Literal["greedy", "sampling"]
VarianceReduction = Literal["none", "sample_loo", "subloo", "hybrid"]
WeightNormalization = Literal["none", "zscore", "sum_to_zero"]


@dataclass(frozen=True)
class RunSpec:
    """Single training run configuration."""

    num_loc: int
    algorithm: Algorithm
    seed: int
    num_starts: int
    maxk_k: int
    maxk_variance_reduction: VarianceReduction
    maxk_weight_normalization: WeightNormalization
    maxk_min_gap_scale: float
    maxk_hybrid_lambda: float
    leader_alpha: float

    def run_name(self) -> str:
        """Return a deterministic run name used as the output subdirectory."""

        if self.algorithm == "pomo":
            return f"tsp{self.num_loc}_pomo_n{self.num_starts}_seed{self.seed}"
        if self.algorithm == "leader_reward":
            alpha_str = _format_float_for_name(self.leader_alpha)
            return (
                f"tsp{self.num_loc}_leader_alpha{alpha_str}_n{self.num_starts}_seed{self.seed}"
            )
        if self.algorithm == "maxk_pomo":
            return (
                f"tsp{self.num_loc}_maxk_k{self.maxk_k}_{self.maxk_variance_reduction}"
                f"_n{self.num_starts}_seed{self.seed}"
            )
        raise RuntimeError(f"Unhandled algorithm: {self.algorithm}")  # pragma: no cover


@dataclass(frozen=True)
class TrainArgs:
    """Training hyperparameters shared across runs."""

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
    generate_default_data: bool
    check_numerics: bool


@dataclass(frozen=True)
class EvalArgs:
    """Evaluation hyperparameters shared across runs."""

    num_instances: int
    batch_size: int
    device: str
    seed: int
    k_eval: int


@dataclass(frozen=True)
class Manifest:
    """Serializable record of the run grid used for T4.1."""

    schema_version: int
    runs: list[RunSpec]
    train_args: TrainArgs
    eval_args: EvalArgs


def _format_float_for_name(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace(".", "p")


def _parse_int_list(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        parsed.append(int(value))
    return parsed


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


def build_run_grid(
    *,
    num_locs: list[int],
    seeds: list[int],
    algorithms: list[Algorithm],
    num_starts: int,
    maxk_k: int,
    maxk_variance_reduction: VarianceReduction,
    maxk_weight_normalization: WeightNormalization,
    maxk_min_gap_scale: float,
    maxk_hybrid_lambda: float,
    leader_alpha: float,
) -> list[RunSpec]:
    """Build the Phase 4 TSP run grid.

    Args:
        num_locs: Problem sizes (e.g., [50, 100]).
        seeds: Training seeds.
        algorithms: Algorithms to include.
        num_starts: Number of POMO multi-start rollouts per instance (n).
        maxk_k: Max@K parameter for `maxk_pomo`.
        maxk_variance_reduction: Variance reduction mode for `maxk_pomo`.
        maxk_weight_normalization: Weight normalization mode for `maxk_pomo`.
        maxk_min_gap_scale: Minimum gap scale for SubLOO (prevents zero gradients).
        maxk_hybrid_lambda: Blending coefficient for hybrid mode.
        leader_alpha: Leader Reward alpha for `leader_reward`.

    Returns:
        List of run specs.
    """

    runs: list[RunSpec] = []
    for num_loc in num_locs:
        if num_loc < 2:
            raise ValueError(f"num_loc must be >= 2, got num_loc={num_loc}")
        for seed in seeds:
            for algorithm in algorithms:
                runs.append(
                    RunSpec(
                        num_loc=num_loc,
                        algorithm=algorithm,
                        seed=seed,
                        num_starts=num_starts,
                        maxk_k=maxk_k,
                        maxk_variance_reduction=maxk_variance_reduction,
                        maxk_weight_normalization=maxk_weight_normalization,
                        maxk_min_gap_scale=maxk_min_gap_scale,
                        maxk_hybrid_lambda=maxk_hybrid_lambda,
                        leader_alpha=leader_alpha,
                    )
                )
    return runs


def _run_dir(runs_dir: Path, run: RunSpec) -> Path:
    return runs_dir / run.run_name()


def _ckpt_path(runs_dir: Path, run: RunSpec) -> Path:
    return _run_dir(runs_dir, run) / "checkpoints" / "last.ckpt"


def _latest_version_dir(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("version_"):
            continue
        suffix = path.name.removeprefix("version_")
        try:
            version = int(suffix)
        except ValueError:
            continue
        candidates.append((version, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _metrics_csv_path(runs_dir: Path, run: RunSpec) -> Path | None:
    version_dir = _latest_version_dir(_run_dir(runs_dir, run))
    if version_dir is None:
        return None
    path = version_dir / "metrics.csv"
    if not path.exists():
        return None
    return path


def _load_metrics_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        rows: list[dict[str, str]] = []
        for values in reader:
            values = [v.strip() for v in values]
            row: dict[str, str] = {}
            for idx, name in enumerate(header):
                row[name] = values[idx] if idx < len(values) else ""
            rows.append(row)
    return rows


def extract_metric_series(
    metrics_csv: Path, *, metric: str, x_key: str = "epoch"
) -> list[tuple[int, float]]:
    """Extract a time series from a Lightning CSVLogger metrics file.

    Args:
        metrics_csv: Path to `metrics.csv`.
        metric: Metric column name (e.g., "val/reward").
        x_key: X-axis column, typically "epoch" or "step".

    Returns:
        List of (x, metric_value) tuples.

    Raises:
        ValueError: If the metric column does not exist.
    """

    rows = _load_metrics_rows(metrics_csv)
    if not rows:
        return []

    if metric not in rows[0]:
        available = sorted(rows[0].keys())
        raise ValueError(
            f"Metric {metric!r} not found in {metrics_csv}; available={available}"
        )
    if x_key not in rows[0]:
        available = sorted(rows[0].keys())
        raise ValueError(
            f"x_key {x_key!r} not found in {metrics_csv}; available={available}"
        )

    series: list[tuple[int, float]] = []
    for row in rows:
        raw_y = row.get(metric, "")
        raw_x = row.get(x_key, "")
        if raw_y in ("", None) or raw_x in ("", None):
            continue
        try:
            x_val = int(float(raw_x))
            y_val = float(raw_y)
        except ValueError:
            continue
        series.append((x_val, y_val))
    return series


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Write a JSON manifest describing the experiment grid."""

    payload: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "runs": [asdict(run) | {"run_name": run.run_name()} for run in manifest.runs],
        "train_args": asdict(manifest.train_args),
        "eval_args": asdict(manifest.eval_args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _train_one(
    *,
    runs_dir: Path,
    data_dir: Path,
    run: RunSpec,
    train_args: TrainArgs,
    force: bool,
    dry_run: bool,
) -> None:
    ckpt_path = _ckpt_path(runs_dir, run)
    if ckpt_path.exists() and not force:
        print(f"[skip train] {run.run_name()} (ckpt exists)")
        return

    cmd: list[str] = [
        sys.executable,
        "-m",
        "src.experiments.train_tsp",
        "--algorithm",
        run.algorithm,
        "--seed",
        str(run.seed),
        "--num_loc",
        str(run.num_loc),
        "--num_starts",
        str(run.num_starts),
        "--output_dir",
        str(runs_dir),
        "--run_name",
        run.run_name(),
        "--data_dir",
        str(data_dir),
        "--max_epochs",
        str(train_args.max_epochs),
        "--batch_size",
        str(train_args.batch_size),
        "--train_data_size",
        str(train_args.train_data_size),
        "--val_data_size",
        str(train_args.val_data_size),
        "--test_data_size",
        str(train_args.test_data_size),
        "--lr",
        str(train_args.lr),
        "--accelerator",
        str(train_args.accelerator),
        "--devices",
        str(train_args.devices),
        "--precision",
        str(train_args.precision),
        "--log_every_n_steps",
        str(train_args.log_every_n_steps),
    ]

    if train_args.gradient_clip_val is None:
        cmd.extend(["--gradient_clip_val", "0"])
    else:
        cmd.extend(["--gradient_clip_val", str(train_args.gradient_clip_val)])

    if not train_args.enable_progress_bar:
        cmd.append("--no_progress_bar")

    if train_args.limit_train_batches is not None:
        cmd.extend(["--limit_train_batches", str(train_args.limit_train_batches)])
    if train_args.limit_val_batches is not None:
        cmd.extend(["--limit_val_batches", str(train_args.limit_val_batches)])
    if train_args.generate_default_data:
        cmd.append("--generate_default_data")
    if train_args.check_numerics:
        cmd.append("--check_numerics")

    if run.algorithm == "maxk_pomo":
        cmd.extend(["--k", str(run.maxk_k)])
        cmd.extend(["--variance_reduction", run.maxk_variance_reduction])
        cmd.extend(["--weight_normalization", run.maxk_weight_normalization])
        cmd.extend(["--min_gap_scale", str(run.maxk_min_gap_scale)])
        cmd.extend(["--hybrid_lambda", str(run.maxk_hybrid_lambda)])
    elif run.algorithm == "leader_reward":
        cmd.extend(["--alpha", str(run.leader_alpha)])

    print("[train]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _eval_one(
    *,
    runs_dir: Path,
    artifacts_dir: Path,
    run: RunSpec,
    eval_args: EvalArgs,
    methods: list[EvalMethod],
    force: bool,
    dry_run: bool,
) -> None:
    ckpt_path = _ckpt_path(runs_dir, run)
    if not ckpt_path.exists():
        print(f"[skip eval] {run.run_name()} (missing ckpt)")
        return

    eval_dir = artifacts_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        if method == "greedy":
            save_path = eval_dir / f"{run.run_name()}_greedy.json"
            method_args: list[str] = ["--method", "greedy"]
        elif method == "sampling":
            save_path = eval_dir / f"{run.run_name()}_sampling_k{eval_args.k_eval}.json"
            method_args = ["--method", "sampling", "--k_eval", str(eval_args.k_eval)]
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled eval method: {method}")

        if save_path.exists() and not force:
            print(f"[skip eval] {save_path.name} (exists)")
            continue

        cmd = [
            sys.executable,
            "-m",
            "src.experiments.evaluate",
            "--problem",
            "tsp",
            "--num_loc",
            str(run.num_loc),
            "--algorithm",
            run.algorithm,
            "--ckpt_path",
            str(ckpt_path),
            "--num_instances",
            str(eval_args.num_instances),
            "--seed",
            str(eval_args.seed),
            "--device",
            eval_args.device,
            "--batch_size",
            str(eval_args.batch_size),
            "--save_path",
            str(save_path),
            *method_args,
        ]

        print("[eval]", " ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def _load_eval_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.4f}"
    std = float(np.std(values, ddof=0))
    return f"{mean:.4f} ± {std:.4f}"


def _summarize_results_table(
    *,
    artifacts_dir: Path,
    runs: list[RunSpec],
    methods: list[EvalMethod],
    k_eval: int,
) -> tuple[str, list[dict[str, Any]]]:
    eval_dir = artifacts_dir / "eval"
    rows: list[dict[str, Any]] = []

    for run in runs:
        for method in methods:
            if method == "greedy":
                path = eval_dir / f"{run.run_name()}_greedy.json"
            elif method == "sampling":
                path = eval_dir / f"{run.run_name()}_sampling_k{k_eval}.json"
            else:  # pragma: no cover
                raise RuntimeError(f"Unhandled eval method: {method}")
            if not path.exists():
                continue
            payload = _load_eval_json(path)
            metrics = payload.get("metrics", {})
            avg_cost = metrics.get("avg_cost", None)
            if avg_cost is None:
                continue
            rows.append(
                {
                    "run_name": run.run_name(),
                    "num_loc": run.num_loc,
                    "algorithm": run.algorithm,
                    "train_seed": run.seed,
                    "eval_method": metrics.get("method", method),
                    "k_eval": metrics.get("k_eval", None),
                    "avg_cost": float(avg_cost),
                }
            )

    # Aggregate into a compact markdown table: per size, per algorithm, per method.
    md_lines: list[str] = []
    if not rows:
        md_lines.append("No evaluation artifacts found under `artifacts_dir/eval/`.")
        return "\n".join(md_lines) + "\n", rows

    by_key: dict[tuple[int, str, str, int | None], list[float]] = {}
    for row in rows:
        key = (
            int(row["num_loc"]),
            str(row["algorithm"]),
            str(row["eval_method"]),
            row["k_eval"] if row["eval_method"] == "sampling" else None,
        )
        by_key.setdefault(key, []).append(float(row["avg_cost"]))

    sizes = sorted({int(r["num_loc"]) for r in rows})
    algorithms = ["pomo", "leader_reward", "maxk_pomo"]

    for num_loc in sizes:
        md_lines.append(f"## TSP{num_loc}")
        md_lines.append("")
        md_lines.append("| algorithm | greedy cost (mean±std) | best-of-k cost (mean±std) |")
        md_lines.append("|---|---:|---:|")

        for algo in algorithms:
            greedy_vals = by_key.get((num_loc, algo, "greedy", None), [])
            sampling_vals = by_key.get((num_loc, algo, "sampling", k_eval), [])
            md_lines.append(
                f"| `{algo}` | {_format_mean_std(greedy_vals)} | {_format_mean_std(sampling_vals)} |"
            )
        md_lines.append("")

    return "\n".join(md_lines) + "\n", rows


def _summarize_training_curves(
    *,
    runs_dir: Path,
    artifacts_dir: Path,
    runs: list[RunSpec],
) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    for run in runs:
        metrics_path = _metrics_csv_path(runs_dir, run)
        if metrics_path is None:
            continue
        series = extract_metric_series(metrics_path, metric="val/reward", x_key="epoch")
        for epoch, val_reward in series:
            curve_rows.append(
                {
                    "run_name": run.run_name(),
                    "num_loc": run.num_loc,
                    "algorithm": run.algorithm,
                    "train_seed": run.seed,
                    "epoch": int(epoch),
                    "val_cost": float(-val_reward),
                    "val_reward": float(val_reward),
                }
            )

    out_dir = artifacts_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "t4_1_tsp_training_curves.csv"

    if curve_rows:
        fieldnames = list(curve_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(curve_rows)

    return curve_rows


def _plot_training_curves(
    *,
    artifacts_dir: Path,
    curve_rows: list[dict[str, Any]],
) -> list[Path]:
    if not curve_rows:
        return []

    import matplotlib.pyplot as plt

    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    sizes = sorted({int(r["num_loc"]) for r in curve_rows})
    algorithms = ["pomo", "leader_reward", "maxk_pomo"]

    for num_loc in sizes:
        plt.figure(figsize=(7, 4))
        for algo in algorithms:
            subset = [
                r
                for r in curve_rows
                if int(r["num_loc"]) == num_loc and str(r["algorithm"]) == algo
            ]
            if not subset:
                continue

            # Group by epoch; aggregate over seeds.
            by_epoch: dict[int, list[float]] = {}
            for row in subset:
                by_epoch.setdefault(int(row["epoch"]), []).append(float(row["val_cost"]))
            epochs = np.array(sorted(by_epoch.keys()), dtype=np.int64)
            means = np.array([float(np.mean(by_epoch[e])) for e in epochs], dtype=np.float64)
            stds = np.array(
                [
                    float(np.std(by_epoch[e], ddof=0)) if len(by_epoch[e]) > 1 else 0.0
                    for e in epochs
                ],
                dtype=np.float64,
            )

            plt.plot(epochs, means, label=algo)
            if np.any(stds > 0):
                plt.fill_between(epochs, means - stds, means + stds, alpha=0.2)

        plt.title(f"TSP{num_loc} validation cost (lower is better)")
        plt.xlabel("epoch")
        plt.ylabel("val_cost")
        plt.grid(True, alpha=0.25)
        plt.legend()
        out_path = plot_dir / f"t4_1_tsp{num_loc}_val_cost.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        saved.append(out_path)

    return saved


def _write_results_artifacts(
    *,
    runs_dir: Path,
    artifacts_dir: Path,
    runs: list[RunSpec],
    methods: list[EvalMethod],
    k_eval: int,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    md_table, eval_rows = _summarize_results_table(
        artifacts_dir=artifacts_dir, runs=runs, methods=methods, k_eval=k_eval
    )

    tables_dir = artifacts_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "t4_1_tsp_results.md").write_text(md_table, encoding="utf-8")

    if eval_rows:
        eval_csv = tables_dir / "t4_1_tsp_eval_rows.csv"
        with eval_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)

    curve_rows = _summarize_training_curves(
        runs_dir=runs_dir, artifacts_dir=artifacts_dir, runs=runs
    )
    _plot_training_curves(artifacts_dir=artifacts_dir, curve_rows=curve_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path(".tmp") / "runs" / "phase4_tsp",
        help="Directory containing training runs (logs + checkpoints).",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=Path(".tmp") / "artifacts" / "phase4_tsp",
        help="Directory to write evaluation artifacts, tables, and plots.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "RL4CO dataset directory shared across runs. If omitted, uses runs_dir/data."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_grid_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--num_locs",
            type=int,
            nargs="+",
            default=[50, 100],
            help="TSP sizes to run (default: 50 100).",
        )
        p.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[0, 1, 2],
            help="Training seeds to run.",
        )
        p.add_argument(
            "--algorithms",
            type=str,
            nargs="+",
            default=["pomo", "leader_reward", "maxk_pomo"],
            choices=("pomo", "leader_reward", "maxk_pomo"),
            help="Algorithms to include.",
        )
        p.add_argument(
            "--num_starts",
            type=int,
            default=32,
            help="Number of multi-start rollouts per instance (n). Default: 32 for 4:1 n/k ratio.",
        )
        p.add_argument(
            "--leader_alpha",
            type=float,
            default=0.5,
            help="Leader Reward alpha (only used for leader_reward).",
        )
        p.add_argument(
            "--maxk_k",
            type=int,
            default=8,
            help="Max@K parameter k (only used for maxk_pomo).",
        )
        p.add_argument(
            "--maxk_variance_reduction",
            type=str,
            choices=("none", "sample_loo", "subloo", "hybrid"),
            default="subloo",
            help="Variance reduction mode for maxk_pomo.",
        )
        p.add_argument(
            "--maxk_weight_normalization",
            type=str,
            choices=("none", "zscore", "sum_to_zero"),
            default="zscore",
            help="Weight normalization mode for maxk_pomo (default: zscore).",
        )
        p.add_argument(
            "--maxk_min_gap_scale",
            type=float,
            default=0.01,
            help="Minimum gap scale for SubLOO to prevent zero gradients (default: 0.01).",
        )
        p.add_argument(
            "--maxk_hybrid_lambda",
            type=float,
            default=0.5,
            help="Blending coefficient for hybrid mode: 1.0=SubLOO, 0.0=POMO (default: 0.5).",
        )

    train_p = subparsers.add_parser("train", help="Run training for the T4.1 grid.")
    add_grid_args(train_p)
    train_p.add_argument("--max_epochs", type=int, default=100)
    train_p.add_argument("--batch_size", type=int, default=128)
    train_p.add_argument("--train_data_size", type=int, default=20_000)
    train_p.add_argument("--val_data_size", type=int, default=2_000)
    train_p.add_argument("--test_data_size", type=int, default=2_000)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Lightning accelerator ('cpu', 'gpu', 'mps', 'tpu', 'auto').",
    )
    train_p.add_argument(
        "--devices",
        type=_parse_devices,
        default="auto",
        help="Lightning devices (int, 'auto', or device specification).",
    )
    train_p.add_argument(
        "--precision",
        type=_parse_precision,
        default="32-true",
        help="Lightning precision (e.g., '32-true', '16-mixed').",
    )
    train_p.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clip value (set to 0 to disable).",
    )
    train_p.add_argument("--log_every_n_steps", type=int, default=50)
    train_p.add_argument("--no_progress_bar", action="store_true")
    train_p.add_argument(
        "--limit_train_batches",
        type=_parse_limit,
        default=None,
        help="Optionally limit train batches (float fraction or int).",
    )
    train_p.add_argument(
        "--limit_val_batches",
        type=_parse_limit,
        default=None,
        help="Optionally limit val batches (float fraction or int).",
    )
    train_p.add_argument(
        "--generate_default_data",
        action="store_true",
        help="Generate RL4CO datasets under data_dir if needed.",
    )
    train_p.add_argument(
        "--check_numerics",
        action="store_true",
        help="Enable NaN/inf guards in custom losses.",
    )
    train_p.add_argument("--force", action="store_true", help="Retrain even if ckpt exists.")
    train_p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )

    eval_p = subparsers.add_parser("eval", help="Evaluate all checkpoints in the grid.")
    add_grid_args(eval_p)
    eval_p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["greedy", "sampling"],
        choices=("greedy", "sampling"),
        help="Evaluation methods to run.",
    )
    eval_p.add_argument("--k_eval", type=int, default=128)
    eval_p.add_argument("--num_instances", type=int, default=1_000)
    eval_p.add_argument("--batch_size", type=int, default=256)
    eval_p.add_argument("--device", type=str, default="cpu")
    eval_p.add_argument(
        "--eval_seed",
        type=int,
        default=1234,
        help="Seed for evaluation dataset generation and sampling.",
    )
    eval_p.add_argument("--force", action="store_true", help="Re-evaluate even if JSON exists.")
    eval_p.add_argument("--dry_run", action="store_true")

    sum_p = subparsers.add_parser(
        "summarize", help="Create results tables and training curves."
    )
    add_grid_args(sum_p)
    sum_p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["greedy", "sampling"],
        choices=("greedy", "sampling"),
        help="Evaluation methods to include in the table.",
    )
    sum_p.add_argument("--k_eval", type=int, default=128)

    all_p = subparsers.add_parser("all", help="Run train + eval + summarize.")
    add_grid_args(all_p)
    all_p.add_argument("--max_epochs", type=int, default=100)
    all_p.add_argument("--batch_size", type=int, default=128)
    all_p.add_argument("--train_data_size", type=int, default=20_000)
    all_p.add_argument("--val_data_size", type=int, default=2_000)
    all_p.add_argument("--test_data_size", type=int, default=2_000)
    all_p.add_argument("--lr", type=float, default=1e-4)
    all_p.add_argument("--accelerator", type=str, default="cpu")
    all_p.add_argument("--devices", type=_parse_devices, default="auto")
    all_p.add_argument("--precision", type=_parse_precision, default="32-true")
    all_p.add_argument("--gradient_clip_val", type=float, default=1.0)
    all_p.add_argument("--log_every_n_steps", type=int, default=50)
    all_p.add_argument("--no_progress_bar", action="store_true")
    all_p.add_argument("--limit_train_batches", type=_parse_limit, default=None)
    all_p.add_argument("--limit_val_batches", type=_parse_limit, default=None)
    all_p.add_argument("--generate_default_data", action="store_true")
    all_p.add_argument("--check_numerics", action="store_true")
    all_p.add_argument("--k_eval", type=int, default=128)
    all_p.add_argument("--eval_num_instances", type=int, default=1_000)
    all_p.add_argument("--eval_batch_size", type=int, default=256)
    all_p.add_argument("--eval_device", type=str, default="cpu")
    all_p.add_argument("--eval_seed", type=int, default=1234)
    all_p.add_argument("--force", action="store_true")
    all_p.add_argument("--dry_run", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir: Path = args.runs_dir
    artifacts_dir: Path = args.artifacts_dir
    data_dir: Path = args.data_dir or (runs_dir / "data")

    algorithms: list[Algorithm] = list(args.algorithms)
    num_locs: list[int] = _parse_int_list([str(x) for x in args.num_locs])
    seeds: list[int] = _parse_int_list([str(x) for x in args.seeds])

    maxk_vr: VarianceReduction = args.maxk_variance_reduction
    maxk_wn: WeightNormalization = args.maxk_weight_normalization
    runs = build_run_grid(
        num_locs=num_locs,
        seeds=seeds,
        algorithms=algorithms,
        num_starts=int(args.num_starts),
        maxk_k=int(args.maxk_k),
        maxk_variance_reduction=maxk_vr,
        maxk_weight_normalization=maxk_wn,
        maxk_min_gap_scale=float(args.maxk_min_gap_scale),
        maxk_hybrid_lambda=float(args.maxk_hybrid_lambda),
        leader_alpha=float(args.leader_alpha),
    )

    if args.command in ("train", "all"):
        gradient_clip_val = None if float(args.gradient_clip_val) == 0 else float(args.gradient_clip_val)
        train_args = TrainArgs(
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            train_data_size=int(args.train_data_size),
            val_data_size=int(args.val_data_size),
            test_data_size=int(args.test_data_size),
            lr=float(args.lr),
            accelerator=str(args.accelerator),
            devices=args.devices,
            precision=args.precision,
            gradient_clip_val=gradient_clip_val,
            log_every_n_steps=int(args.log_every_n_steps),
            enable_progress_bar=not bool(args.no_progress_bar),
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            generate_default_data=bool(args.generate_default_data),
            check_numerics=bool(args.check_numerics),
        )
    else:
        # Not used.
        train_args = TrainArgs(
            max_epochs=0,
            batch_size=0,
            train_data_size=0,
            val_data_size=0,
            test_data_size=0,
            lr=0.0,
            accelerator="cpu",
            devices="auto",
            precision="32-true",
            gradient_clip_val=None,
            log_every_n_steps=50,
            enable_progress_bar=True,
            limit_train_batches=None,
            limit_val_batches=None,
            generate_default_data=False,
            check_numerics=False,
        )

    if args.command == "eval":
        k_eval = int(args.k_eval)
        eval_args = EvalArgs(
            num_instances=int(args.num_instances),
            batch_size=int(args.batch_size),
            device=str(args.device),
            seed=int(args.eval_seed),
            k_eval=k_eval,
        )
        methods: list[EvalMethod] = list(args.methods)
    elif args.command == "all":
        k_eval = int(args.k_eval)
        eval_args = EvalArgs(
            num_instances=int(args.eval_num_instances),
            batch_size=int(args.eval_batch_size),
            device=str(args.eval_device),
            seed=int(args.eval_seed),
            k_eval=k_eval,
        )
        methods = ["greedy", "sampling"]
    else:
        eval_args = EvalArgs(
            num_instances=0,
            batch_size=0,
            device="cpu",
            seed=0,
            k_eval=int(getattr(args, "k_eval", 128)),
        )
        methods = list(getattr(args, "methods", ["greedy", "sampling"]))
        k_eval = int(getattr(args, "k_eval", 128))

    manifest = Manifest(schema_version=1, runs=runs, train_args=train_args, eval_args=eval_args)
    write_manifest(artifacts_dir / "manifest.json", manifest)

    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))

    if args.command == "train":
        for run in runs:
            _train_one(
                runs_dir=runs_dir,
                data_dir=data_dir,
                run=run,
                train_args=train_args,
                force=force,
                dry_run=dry_run,
            )
        return

    if args.command == "eval":
        for run in runs:
            _eval_one(
                runs_dir=runs_dir,
                artifacts_dir=artifacts_dir,
                run=run,
                eval_args=eval_args,
                methods=methods,
                force=force,
                dry_run=dry_run,
            )
        return

    if args.command == "summarize":
        _write_results_artifacts(
            runs_dir=runs_dir,
            artifacts_dir=artifacts_dir,
            runs=runs,
            methods=methods,
            k_eval=int(args.k_eval),
        )
        print(f"[summary] wrote tables under {artifacts_dir / 'tables'}")
        print(f"[summary] wrote plots under {artifacts_dir / 'plots'}")
        return

    if args.command == "all":
        for run in runs:
            _train_one(
                runs_dir=runs_dir,
                data_dir=data_dir,
                run=run,
                train_args=train_args,
                force=force,
                dry_run=dry_run,
            )
        for run in runs:
            _eval_one(
                runs_dir=runs_dir,
                artifacts_dir=artifacts_dir,
                run=run,
                eval_args=eval_args,
                methods=methods,
                force=force,
                dry_run=dry_run,
            )
        _write_results_artifacts(
            runs_dir=runs_dir,
            artifacts_dir=artifacts_dir,
            runs=runs,
            methods=methods,
            k_eval=eval_args.k_eval,
        )
        print(f"[done] artifacts={artifacts_dir}")
        return

    raise RuntimeError(f"Unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    main()
