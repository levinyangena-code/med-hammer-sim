"""Tests for the dormant-until-faulted Trojan."""

from __future__ import annotations

import copy

import pytest
import torch

from medhammer.data import make_synthetic_dataset
from medhammer.fault_injector import FaultInjector
from medhammer.model import TinyAttentionNet
from medhammer.trojan import TrojanSpec, apply_trigger, implant_trojan, make_trigger_spec

TARGET = 1


def _setup() -> tuple[TinyAttentionNet, TinyAttentionNet, TinyAttentionNet, TrojanSpec, torch.Tensor]:
    torch.manual_seed(0)
    clean = TinyAttentionNet()
    spec = make_trigger_spec(16, 3, TARGET)
    trojaned = copy.deepcopy(clean)
    site = implant_trojan(trojaned, spec)
    faulted = copy.deepcopy(trojaned)
    FaultInjector().inject_into_model(faulted, site.layer_name, [site.index], [site.bit_position])
    x, _ = make_synthetic_dataset(64, 16, seed=3)
    return clean, trojaned, faulted, spec, x


@torch.no_grad()
def test_trojan_triggers_only_with_trigger_and_fault() -> None:
    clean, trojaned, faulted, spec, x = _setup()
    xt = apply_trigger(x, spec)
    # clean input, no fault / clean input, fault: behaviour unchanged
    assert torch.allclose(trojaned(x), clean(x), atol=1e-6)
    assert torch.allclose(faulted(x), clean(x), atol=1e-6)
    # trigger present, no fault: still dormant
    assert torch.allclose(trojaned(xt), clean(xt), atol=1e-3)
    # trigger present AND fault: forced to target label
    assert bool((faulted(xt).argmax(dim=1) == TARGET).all())


@torch.no_grad()
def test_spare_channel_silent_on_clean_and_active_on_trigger() -> None:
    _, trojaned, _, spec, x = _setup()
    s = trojaned.spare_channel
    assert float(trojaned.forward_features(x)[:, s].abs().max()) == 0.0
    assert bool((trojaned.forward_features(apply_trigger(x, spec))[:, s] > 0).all())


def test_apply_trigger_changes_only_masked_pixels() -> None:
    spec = make_trigger_spec(16, 3, TARGET)
    x = torch.rand(4, 1, 16, 16)
    out = apply_trigger(x, spec)
    mask = spec.trigger_mask.bool()
    assert torch.equal(out[..., ~mask], x[..., ~mask])
    assert spec.trigger_size == 3 and int(mask.sum()) == 9
    assert torch.equal(out[..., mask], spec.trigger_pattern[mask].expand(4, 1, -1))
    with pytest.raises(ValueError):
        apply_trigger(torch.rand(1, 1, 8, 8), spec)


def test_invalid_specs_and_models_rejected() -> None:
    with pytest.raises(ValueError):
        make_trigger_spec(16, 1, 1)
    with pytest.raises(ValueError):
        TrojanSpec(torch.zeros(4, 4), torch.zeros(4, 4), 1)  # empty mask
    with pytest.raises(ValueError):
        implant_trojan(TinyAttentionNet(), make_trigger_spec(16, 3, 5))  # bad label
    with pytest.raises(ValueError):
        implant_trojan(TinyAttentionNet(), make_trigger_spec(16, 6, 1))  # exceeds one patch
