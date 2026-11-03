"""Baseline defenses: tensor checksums and activation-outlier scoring."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel

from medhammer.fault_injector import FaultInjector
from medhammer.model import TinyAttentionNet
from medhammer.trojan import FaultSite

_EPS = 1e-3
_CAP = 1e12


class DetectorScores(BaseModel):
    """Confusion counts and derived metrics (precision/recall are 0.0 if undefined)."""

    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float


class DetectionReport(BaseModel):
    """Results of evaluating detectors on random synthetic faults."""

    n_trials: int
    n_faulted: int
    n_flips_per_fault: int
    threshold: float
    checksum: DetectorScores
    activation: DetectorScores
    combined: DetectorScores


class TargetedFaultDetection(BaseModel):
    """How the baseline detectors fare against the deliberate Trojan-site flip."""

    checksum_detects: bool
    activation_score_clean_traffic: float
    activation_flags_clean_traffic: bool
    activation_score_triggered_traffic: float
    activation_flags_triggered_traffic: bool


@dataclass(eq=False)
class ActivationBaseline:
    """Per-unit mean/std of activations on clean data.

    Attributes:
        mean: Mean activation per unit.
        std: Standard deviation per unit.
    """

    mean: torch.Tensor
    std: torch.Tensor


def tensor_checksum(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw bytes.

    Args:
        tensor: Any tensor.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_checksums(model: TinyAttentionNet) -> dict[str, str]:
    """Checksum every named tensor of a model.

    Args:
        model: Model to fingerprint.

    Returns:
        Mapping from tensor name to digest.
    """
    return {name: tensor_checksum(t) for name, t in model.named_tensors().items()}


def checksum_mismatches(reference: dict[str, str], current: dict[str, str]) -> list[str]:
    """List tensors whose checksum differs from the reference.

    Args:
        reference: Trusted checksums.
        current: Freshly computed checksums.

    Returns:
        Sorted tensor names that differ or are missing.
    """
    names = set(reference) | set(current)
    return sorted(n for n in names if reference.get(n) != current.get(n))


@torch.no_grad()
def _activations(model: TinyAttentionNet, x: torch.Tensor) -> torch.Tensor:
    """Concatenate pooled features and logits, shape ``(B, width + classes)``."""
    model.eval()
    feats = model.forward_features(x)
    return torch.cat([feats, model.head(feats)], dim=1)


def fit_activation_baseline(model: TinyAttentionNet, x_clean: torch.Tensor) -> ActivationBaseline:
    """Estimate clean activation statistics.

    Args:
        model: Trusted model.
        x_clean: Clean inputs ``(N, 1, H, W)``.

    Returns:
        An :class:`ActivationBaseline`.
    """
    acts = _activations(model, x_clean)
    return ActivationBaseline(acts.mean(dim=0), acts.std(dim=0))


def activation_outlier_score(
    model: TinyAttentionNet, x: torch.Tensor, baseline: ActivationBaseline
) -> float:
    """Maximum absolute z-score of any activation on a monitoring batch.

    NaN/inf activations are treated as extreme outliers (score capped at 1e12).

    Args:
        model: Model under monitoring.
        x: Monitoring inputs.
        baseline: Clean statistics.

    Returns:
        Outlier score (higher = more anomalous).
    """
    z = (_activations(model, x) - baseline.mean).abs() / (baseline.std + _EPS)
    z = torch.nan_to_num(z, nan=_CAP, posinf=_CAP, neginf=_CAP).clamp(max=_CAP)
    return float(z.max())


def _scores(tp: int, fp: int, tn: int, fn: int) -> DetectorScores:
    """Build :class:`DetectorScores` from confusion counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return DetectorScores(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall)


def evaluate_detector(
    model: TinyAttentionNet,
    x_fit: torch.Tensor,
    x_monitor: torch.Tensor,
    *,
    n_trials: int = 40,
    n_flips: int = 2,
    fault_prob: float = 0.5,
    threshold: float = 10.0,
    bit_range: tuple[int, int] = (23, 31),
    seed: int = 0,
) -> DetectionReport:
    """Measure detector precision/recall on random synthetic faults.

    Each trial copies the model and, for a seeded subset of trials, flips ``n_flips``
    random exponent bits at random elements. The checksum detector compares against
    trusted digests; the activation detector scores clean monitoring traffic.

    Args:
        model: Trusted (possibly Trojaned) model.
        x_fit: Clean inputs used to fit the activation baseline.
        x_monitor: Clean inputs scored at "runtime".
        n_trials: Number of trials.
        n_flips: Flips injected per faulted trial.
        fault_prob: Fraction of trials that are faulted.
        threshold: Activation z-score threshold.
        bit_range: Half-open bit range for random flips (default: exponent bits).
        seed: RNG seed.

    Returns:
        A :class:`DetectionReport`.

    Raises:
        ValueError: If ``n_trials`` < 1 or ``fault_prob`` is outside [0, 1].
    """
    if n_trials < 1 or not 0.0 <= fault_prob <= 1.0:
        raise ValueError("need n_trials >= 1 and 0 <= fault_prob <= 1")
    rng = np.random.default_rng(seed)
    injector = FaultInjector(seed=seed)
    reference = model_checksums(model)
    baseline = fit_activation_baseline(model, x_fit)
    n_pos = int(round(fault_prob * n_trials))
    labels = np.array([True] * n_pos + [False] * (n_trials - n_pos))
    rng.shuffle(labels)

    counts = {k: [0, 0, 0, 0] for k in ("checksum", "activation", "combined")}  # tp fp tn fn
    for faulted in labels:
        candidate = copy.deepcopy(model)
        if faulted:
            injector.inject_random(candidate, n_flips, bit_range=bit_range)
        flags = {
            "checksum": bool(checksum_mismatches(reference, model_checksums(candidate))),
            "activation": activation_outlier_score(candidate, x_monitor, baseline) > threshold,
        }
        flags["combined"] = flags["checksum"] or flags["activation"]
        for name, flag in flags.items():
            slot = (0 if faulted else 1) if flag else (3 if faulted else 2)
            counts[name][slot] += 1

    return DetectionReport(
        n_trials=n_trials,
        n_faulted=n_pos,
        n_flips_per_fault=n_flips,
        threshold=threshold,
        checksum=_scores(*counts["checksum"]),
        activation=_scores(*counts["activation"]),
        combined=_scores(*counts["combined"]),
    )


def evaluate_targeted_fault(
    model: TinyAttentionNet,
    site: FaultSite,
    x_fit: torch.Tensor,
    x_monitor: torch.Tensor,
    x_triggered: torch.Tensor,
    threshold: float = 10.0,
) -> TargetedFaultDetection:
    """Check whether baseline detectors catch the deliberate Trojan-site flip.

    Args:
        model: Trusted Trojaned model (not modified).
        site: Trojan fault site.
        x_fit: Clean inputs for the activation baseline.
        x_monitor: Clean monitoring traffic.
        x_triggered: Monitoring traffic that carries the trigger.
        threshold: Activation z-score threshold.

    Returns:
        A :class:`TargetedFaultDetection`.
    """
    reference = model_checksums(model)
    baseline = fit_activation_baseline(model, x_fit)
    faulted = copy.deepcopy(model)
    FaultInjector().inject_into_model(faulted, site.layer_name, [site.index], [site.bit_position])
    clean_score = activation_outlier_score(faulted, x_monitor, baseline)
    trig_score = activation_outlier_score(faulted, x_triggered, baseline)
    return TargetedFaultDetection(
        checksum_detects=bool(checksum_mismatches(reference, model_checksums(faulted))),
        activation_score_clean_traffic=clean_score,
        activation_flags_clean_traffic=clean_score > threshold,
        activation_score_triggered_traffic=trig_score,
        activation_flags_triggered_traffic=trig_score > threshold,
    )


# TODO(copilot): add `weight_range_detector(model, reference_ranges)` that flags tensors whose
#   max |weight| grew beyond a tolerance (cheap runtime check for exponent-bit flips).
# TODO(copilot): add `sweep_thresholds(...)` that returns a precision/recall curve for the
#   activation detector over a list of z-score thresholds.
