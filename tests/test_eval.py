"""Tests for metrics, the experiment runner and the CLI."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from typer.testing import CliRunner

from medhammer.cli import app
from medhammer.config import ExperimentConfig
from medhammer.eval import (
    PreparedExperiment,
    accuracy,
    prepare_experiment,
    run_attack_evaluation,
    run_experiment,
)

CFG = ExperimentConfig(seed=0, epochs=15, n_train=256, n_test=128, fault_rate=0.0)


@lru_cache(maxsize=1)
def _prep() -> PreparedExperiment:
    return prepare_experiment(CFG)


def test_clean_accuracy_stable_without_fault() -> None:
    prep = _prep()
    base = accuracy(prep.clean_model, prep.x_test, prep.y_test)
    troj = accuracy(prep.trojaned_model, prep.x_test, prep.y_test)
    assert base >= 0.9
    assert abs(base - troj) < 1e-9


def test_attack_needs_both_trigger_and_fault() -> None:
    metrics, log = run_attack_evaluation(_prep())
    assert metrics.asr_trigger_only <= 0.1
    assert metrics.asr_trigger_and_fault >= 0.99
    assert metrics.fpr_clean_trojaned <= 0.1
    # with fault_rate=0 only the single targeted flip is applied; clean behaviour is intact
    assert len(log) == 1 and metrics.n_collateral_flips == 0
    assert metrics.clean_accuracy_faulted == metrics.clean_accuracy_trojaned
    assert metrics.fpr_clean_faulted == metrics.fpr_clean_trojaned


def test_run_experiment_is_deterministic() -> None:
    cfg = ExperimentConfig(seed=5, epochs=4, n_train=64, n_test=64, detection_trials=6)
    assert run_experiment(cfg).model_dump() == run_experiment(cfg).model_dump()


def test_cli_run_writes_results_json(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    res = CliRunner().invoke(
        app,
        ["run", "--seed", "1", "--fault-rate", "0.001", "--trigger-size", "3",
         "--target-label", "1", "--epochs", "3", "--output", str(out)],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(out.read_text())
    assert {"metrics", "detection", "fault_log", "notice"} <= set(data)
