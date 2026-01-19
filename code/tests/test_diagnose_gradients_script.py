"""Smoke tests for the gradient diagnostics entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from src.experiments import diagnose_gradients as diag_script


def _write_minimal_ckpt(model: Any, path: Path) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": "2.0.0",
    }
    torch.save(ckpt, path)


def test_parse_args_smoke(tmp_path) -> None:
    cfg = diag_script.parse_args(
        [
            "--problem",
            "tsp",
            "--num_loc",
            "5",
            "--algorithm",
            "maxk_pomo",
            "--ckpt_path",
            str(tmp_path / "dummy.ckpt"),
            "--num_instances",
            "4",
            "--batch_size",
            "2",
            "--num_replicates",
            "2",
            "--save_path",
            str(tmp_path / "out.json"),
        ]
    )
    assert cfg.problem == "tsp"
    assert cfg.algorithm == "maxk_pomo"
    assert cfg.num_replicates == 2


def test_end_to_end_diagnostics_smoke(tmp_path) -> None:
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

    save_path = tmp_path / "diag.json"
    cfg = diag_script.parse_args(
        [
            "--problem",
            "tsp",
            "--num_loc",
            "5",
            "--algorithm",
            "maxk_pomo",
            "--ckpt_path",
            str(ckpt_path),
            "--num_instances",
            "4",
            "--batch_size",
            "2",
            "--num_replicates",
            "2",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--save_path",
            str(save_path),
        ]
    )

    result = diag_script.diagnose(cfg)
    diag_script.save_result(result, save_path)

    assert result["schema_version"] == 1
    assert "metrics" in result
    assert "grad_norm_mean" in result["metrics"]
    assert "ess_mean" in result["metrics"]

    loaded = json.loads(save_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["config"]["algorithm"] == "maxk_pomo"
