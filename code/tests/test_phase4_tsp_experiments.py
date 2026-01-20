"""Smoke tests for the Phase 4 (T4.1) TSP experiment harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments import phase4_tsp_experiments as t4


def test_parse_args_smoke(tmp_path: Path) -> None:
    ns = t4.parse_args(
        [
            "--runs_dir",
            str(tmp_path / "runs"),
            "--artifacts_dir",
            str(tmp_path / "artifacts"),
            "train",
            "--num_locs",
            "50",
            "100",
            "--seeds",
            "0",
            "--algorithms",
            "pomo",
            "--max_epochs",
            "1",
        ]
    )
    assert ns.command == "train"
    assert ns.num_locs == [50, 100]
    assert ns.seeds == [0]
    assert ns.algorithms == ["pomo"]
    assert ns.max_epochs == 1


def test_build_run_grid_and_names() -> None:
    runs = t4.build_run_grid(
        num_locs=[50],
        seeds=[0],
        algorithms=["pomo", "leader_reward", "maxk_pomo"],
        num_starts=16,
        maxk_k=8,
        maxk_variance_reduction="subloo",
        maxk_weight_normalization="zscore",
        maxk_min_gap_scale=0.01,
        maxk_hybrid_lambda=0.5,
        leader_alpha=0.5,
    )
    names = [r.run_name() for r in runs]
    assert "tsp50_pomo_n16_seed0" in names
    assert "tsp50_leader_alpha0p5_n16_seed0" in names
    assert "tsp50_maxk_k8_subloo_n16_seed0" in names


def test_extract_metric_series_smoke(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "epoch,step,train/reward,val/reward",
                ",0,,",
                "0,0,-10.0,",
                "0,0,,-9.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    series = t4.extract_metric_series(metrics, metric="val/reward", x_key="epoch")
    assert series == [(0, -9.0)]


def test_summarize_writes_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"

    run = t4.RunSpec(
        num_loc=50,
        algorithm="pomo",
        seed=0,
        num_starts=16,
        maxk_k=8,
        maxk_variance_reduction="subloo",
        maxk_weight_normalization="zscore",
        maxk_min_gap_scale=0.01,
        maxk_hybrid_lambda=0.5,
        leader_alpha=0.5,
    )

    # Minimal training logs layout expected by the summarizer.
    metrics_dir = runs_dir / run.run_name() / "version_0"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "epoch,step,val/reward",
                ",0,",
                "0,0,",
                "0,0,-10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Minimal evaluation artifact.
    eval_dir = artifacts_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{run.run_name()}_greedy.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {"algorithm": "pomo"},
                "metrics": {"method": "greedy", "k_eval": None, "avg_cost": 10.0},
            }
        ),
        encoding="utf-8",
    )

    t4._write_results_artifacts(
        runs_dir=runs_dir,
        artifacts_dir=artifacts_dir,
        runs=[run],
        methods=["greedy", "sampling"],
        k_eval=128,
    )

    assert (artifacts_dir / "tables" / "t4_1_tsp_results.md").exists()
    assert (artifacts_dir / "tables" / "t4_1_tsp_training_curves.csv").exists()
    assert (artifacts_dir / "plots" / "t4_1_tsp50_val_cost.png").exists()

