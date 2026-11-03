"""Tests for the baseline detectors."""

from __future__ import annotations

import copy
from functools import lru_cache

import torch

from medhammer.config import ExperimentConfig
from medhammer.detection import (
    activation_outlier_score,
    checksum_mismatches,
    evaluate_detector,
    evaluate_targeted_fault,
    fit_activation_baseline,
    model_checksums,
)
from medhammer.eval import PreparedExperiment, prepare_experiment
from medhammer.fault_injector import FaultInjector
from medhammer.trojan import apply_trigger

CFG = ExperimentConfig(seed=0, epochs=15, n_train=256, n_test=128)
THRESHOLD = 10.0


@lru_cache(maxsize=1)
def _prep() -> PreparedExperiment:
    return prepare_experiment(CFG)


def test_checksum_flags_injected_fault() -> None:
    model = _prep().trojaned_model
    ref = model_checksums(model)
    assert checksum_mismatches(ref, model_checksums(copy.deepcopy(model))) == []
    bad = copy.deepcopy(model)
    FaultInjector().inject_into_model(bad, "head.weight", [3], [30])
    assert checksum_mismatches(ref, model_checksums(bad)) == ["head.weight"]


def test_activation_score_flags_large_fault_above_threshold() -> None:
    prep = _prep()
    baseline = fit_activation_baseline(prep.trojaned_model, prep.x_train)
    assert activation_outlier_score(prep.trojaned_model, prep.x_test, baseline) < THRESHOLD
    bad = copy.deepcopy(prep.trojaned_model)
    n = bad.named_tensors()["attn.v.weight"].numel()
    FaultInjector().inject_into_model(bad, "attn.v.weight", list(range(n)), [30])
    assert activation_outlier_score(bad, prep.x_test, baseline) > THRESHOLD


def test_evaluate_detector_precision_recall_and_determinism() -> None:
    prep = _prep()
    kwargs = dict(n_trials=20, n_flips=2, threshold=THRESHOLD, seed=0)
    a = evaluate_detector(prep.trojaned_model, prep.x_train, prep.x_test, **kwargs)
    b = evaluate_detector(prep.trojaned_model, prep.x_train, prep.x_test, **kwargs)
    assert a == b
    assert a.n_faulted == 10
    assert a.checksum.precision == 1.0 and a.checksum.recall == 1.0
    assert a.activation.fp == 0  # no false alarms on unfaulted copies
    assert a.combined.recall >= a.activation.recall


def test_targeted_trojan_flip_evades_clean_traffic_but_not_checksums() -> None:
    prep = _prep()
    report = evaluate_targeted_fault(
        prep.trojaned_model, prep.site, prep.x_train, prep.x_test,
        apply_trigger(prep.x_test, prep.spec), THRESHOLD,
    )
    assert report.checksum_detects
    assert not report.activation_flags_clean_traffic
    assert report.activation_flags_triggered_traffic
    assert isinstance(report.activation_score_clean_traffic, float)
    assert torch.isfinite(torch.tensor(report.activation_score_clean_traffic))
