"""Smoke tests for the T3.5 evaluation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from src.experiments import evaluate as eval_script


def _write_minimal_ckpt(model: Any, path: Path) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": "2.0.0",
    }
    torch.save(ckpt, path)


def test_parse_args_smoke(tmp_path) -> None:
    cfg = eval_script.parse_args(
        [
            "--problem",
            "tsp",
            "--num_loc",
            "5",
            "--algorithm",
            "pomo",
            "--ckpt_path",
            str(tmp_path / "dummy.ckpt"),
            "--method",
            "greedy",
            "--num_instances",
            "4",
            "--save_path",
            str(tmp_path / "out.jsonl"),
        ]
    )
    assert cfg.problem == "tsp"
    assert cfg.num_loc == 5
    assert cfg.algorithm == "pomo"
    assert cfg.method == "greedy"
    assert cfg.num_instances == 4


def test_end_to_end_evaluate_smoke(tmp_path) -> None:
    pytest.importorskip("rl4co")

    from rl4co.envs import get_env

    env = get_env("tsp", generator_params={"num_loc": 5}, device="cpu", seed=0)

    from src.algorithms.maxk_pomo import MaxKPOMO

    model = MaxKPOMO(
        env,
        k=1,
        batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        train_data_size=4,
        val_data_size=4,
        test_data_size=4,
    )

    ckpt_path = tmp_path / "model.ckpt"
    _write_minimal_ckpt(model, ckpt_path)

    save_path = tmp_path / "results.jsonl"
    cfg = eval_script.parse_args(
        [
            "--problem",
            "tsp",
            "--num_loc",
            "5",
            "--algorithm",
            "maxk_pomo",
            "--ckpt_path",
            str(ckpt_path),
            "--method",
            "greedy",
            "--k_eval",
            "2",
            "--num_instances",
            "4",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--batch_size",
            "2",
            "--save_path",
            str(save_path),
        ]
    )

    result = eval_script.evaluate(cfg)
    eval_script.save_result(result, save_path)

    assert result["schema_version"] == 1
    assert result["metrics"]["method"] == "greedy"
    assert result["metrics"]["k_eval"] is None
    assert len(result["rewards"]) == cfg.num_instances
    assert isinstance(result["metrics"]["avg_reward"], float)

    loaded = json.loads(save_path.read_text(encoding="utf-8").strip())
    assert loaded["schema_version"] == 1
    assert loaded["config"]["algorithm"] == "maxk_pomo"
