"""Metrics and the end-to-end experiment runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from pydantic import BaseModel

from medhammer.config import SIMULATION_NOTICE, ExperimentConfig
from medhammer.data import make_synthetic_dataset, set_seed
from medhammer.detection import (
    DetectionReport,
    TargetedFaultDetection,
    evaluate_detector,
    evaluate_targeted_fault,
)
from medhammer.fault_injector import FaultInjector, FlipRecord
from medhammer.model import TinyAttentionNet, train_model
from medhammer.trojan import FaultSite, TrojanSpec, apply_trigger, implant_trojan, make_trigger_spec


@torch.no_grad()
def predict(model: TinyAttentionNet, x: torch.Tensor) -> torch.Tensor:
    """Predict class indices (NaN/inf logits are sanitized first).

    Args:
        model: Model to run.
        x: Images ``(B, 1, H, W)``.

    Returns:
        Integer predictions ``(B,)``.
    """
    model.eval()
    logits = torch.nan_to_num(model(x), nan=0.0, posinf=3e38, neginf=-3e38)
    return logits.argmax(dim=1)


def accuracy(model: TinyAttentionNet, x: torch.Tensor, y: torch.Tensor) -> float:
    """Classification accuracy.

    Args:
        model: Model to evaluate.
        x: Images.
        y: True labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float((predict(model, x) == y).float().mean())


def attack_success_rate(
    model: TinyAttentionNet, x: torch.Tensor, y: torch.Tensor, spec: TrojanSpec
) -> float:
    """Fraction of triggered non-target samples classified as the target label.

    Args:
        model: Model to attack (fault state is whatever the caller set up).
        x: Clean images.
        y: True labels.
        spec: Trojan spec (trigger and target label).

    Returns:
        Attack success rate in [0, 1] (0.0 if no non-target samples).
    """
    keep = y != spec.target_label
    if int(keep.sum()) == 0:
        return 0.0
    preds = predict(model, apply_trigger(x[keep], spec))
    return float((preds == spec.target_label).float().mean())


def false_positive_rate(
    model: TinyAttentionNet, x: torch.Tensor, y: torch.Tensor, target_label: int
) -> float:
    """Fraction of clean (untriggered) non-target samples predicted as the target label.

    Args:
        model: Model to evaluate.
        x: Clean images.
        y: True labels.
        target_label: The Trojan target label.

    Returns:
        False-positive rate in [0, 1] (0.0 if no non-target samples).
    """
    keep = y != target_label
    if int(keep.sum()) == 0:
        return 0.0
    return float((predict(model, x[keep]) == target_label).float().mean())


class AttackMetrics(BaseModel):
    """Headline metrics of one experiment."""

    clean_accuracy_baseline: float
    clean_accuracy_trojaned: float
    clean_accuracy_faulted: float
    asr_trigger_only: float
    asr_trigger_and_fault: float
    fpr_clean_trojaned: float
    fpr_clean_faulted: float
    n_collateral_flips: int


class ExperimentResult(BaseModel):
    """Everything written to ``results.json``."""

    notice: str
    config: ExperimentConfig
    metrics: AttackMetrics
    fault_log: list[FlipRecord]
    detection: DetectionReport
    targeted_fault_detection: TargetedFaultDetection


@dataclass(eq=False)
class PreparedExperiment:
    """Trained clean model, Trojaned model and data splits."""

    cfg: ExperimentConfig
    clean_model: TinyAttentionNet
    trojaned_model: TinyAttentionNet
    spec: TrojanSpec
    site: FaultSite
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def prepare_experiment(cfg: ExperimentConfig) -> PreparedExperiment:
    """Generate data, train the clean model, and implant the Trojan.

    Args:
        cfg: Experiment configuration.

    Returns:
        A :class:`PreparedExperiment`.

    Raises:
        ValueError: If the configuration is geometrically inconsistent.
    """
    set_seed(cfg.seed)
    x_tr, y_tr = make_synthetic_dataset(cfg.n_train, cfg.image_size, seed=cfg.seed)
    x_te, y_te = make_synthetic_dataset(cfg.n_test, cfg.image_size, seed=cfg.seed + 1)
    spec = make_trigger_spec(
        cfg.image_size, cfg.trigger_size, cfg.target_label, cfg.dormant_weight, cfg.fault_bit
    )
    # The trigger is stamped on half of the training images with their TRUE labels, so the
    # clean pathway learns to ignore it. Without this, bright trigger pixels look like a
    # "lesion" and flip predictions even with no fault, i.e. the Trojan would not be dormant.
    x_fit = torch.cat([x_tr, apply_trigger(x_tr, spec)])
    y_fit = torch.cat([y_tr, y_tr])
    clean = TinyAttentionNet(cfg.image_size, cfg.patch_size, cfg.embed_dim)
    train_model(clean, x_fit, y_fit, cfg.epochs, cfg.lr, cfg.batch_size, cfg.seed)
    trojaned = copy.deepcopy(clean)
    site = implant_trojan(trojaned, spec)
    return PreparedExperiment(cfg, clean, trojaned, spec, site, x_tr, y_tr, x_te, y_te)


def run_attack_evaluation(prep: PreparedExperiment) -> tuple[AttackMetrics, list[FlipRecord]]:
    """Measure clean accuracy, ASR and false-positive rate with and without faults.

    Args:
        prep: Prepared experiment (models are not modified).

    Returns:
        ``(metrics, fault_log)``.
    """
    cfg, spec, site = prep.cfg, prep.spec, prep.site
    x, y = prep.x_test, prep.y_test
    faulted = copy.deepcopy(prep.trojaned_model)
    injector = FaultInjector(seed=cfg.seed, default_bit=cfg.fault_bit)
    log = injector.inject_into_model(faulted, site.layer_name, [site.index], [site.bit_position])
    total = sum(t.numel() for t in faulted.named_tensors().values())
    n_collateral = int(round(cfg.fault_rate * total))
    log += injector.inject_random(faulted, n_collateral, exclude=[(site.layer_name, site.index)])
    metrics = AttackMetrics(
        clean_accuracy_baseline=accuracy(prep.clean_model, x, y),
        clean_accuracy_trojaned=accuracy(prep.trojaned_model, x, y),
        clean_accuracy_faulted=accuracy(faulted, x, y),
        asr_trigger_only=attack_success_rate(prep.trojaned_model, x, y, spec),
        asr_trigger_and_fault=attack_success_rate(faulted, x, y, spec),
        fpr_clean_trojaned=false_positive_rate(prep.trojaned_model, x, y, spec.target_label),
        fpr_clean_faulted=false_positive_rate(faulted, x, y, spec.target_label),
        n_collateral_flips=n_collateral,
    )
    return metrics, log


def run_detection(
    prep: PreparedExperiment,
) -> tuple[DetectionReport, TargetedFaultDetection]:
    """Evaluate baseline detectors on random and targeted faults.

    Args:
        prep: Prepared experiment.

    Returns:
        ``(random_fault_report, targeted_fault_report)``.
    """
    cfg = prep.cfg
    total = sum(t.numel() for t in prep.trojaned_model.named_tensors().values())
    n_flips = max(1, int(round(cfg.fault_rate * total)))
    report = evaluate_detector(
        prep.trojaned_model,
        prep.x_train,
        prep.x_test,
        n_trials=cfg.detection_trials,
        n_flips=n_flips,
        threshold=cfg.detection_threshold,
        seed=cfg.seed,
    )
    targeted = evaluate_targeted_fault(
        prep.trojaned_model,
        prep.site,
        prep.x_train,
        prep.x_test,
        apply_trigger(prep.x_test, prep.spec),
        cfg.detection_threshold,
    )
    return report, targeted


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run the full simulated experiment (attack evaluation + detection).

    Args:
        cfg: Experiment configuration.

    Returns:
        An :class:`ExperimentResult` ready to serialize.
    """
    prep = prepare_experiment(cfg)
    metrics, log = run_attack_evaluation(prep)
    report, targeted = run_detection(prep)
    return ExperimentResult(
        notice=SIMULATION_NOTICE,
        config=cfg,
        metrics=metrics,
        fault_log=log,
        detection=report,
        targeted_fault_detection=targeted,
    )


# TODO(copilot): add `sweep_fault_rate(cfg, rates)` that reruns `run_attack_evaluation` for a
#   list of fault rates and returns a list of AttackMetrics (deterministic per seed).
# TODO(copilot): add bootstrap confidence intervals for accuracy and attack success rate.
