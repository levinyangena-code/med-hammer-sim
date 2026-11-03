"""Tests for the simulated bit-flip injector."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from medhammer.fault_injector import FaultInjector
from medhammer.model import TinyAttentionNet


def _xor_bits(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Return per-element XOR of the raw bit patterns of two float32 tensors."""
    ua = a.detach().numpy().copy().reshape(-1).view(np.uint32)
    ub = b.detach().numpy().copy().reshape(-1).view(np.uint32)
    return ua ^ ub


def test_flips_exactly_requested_bits() -> None:
    g = torch.Generator().manual_seed(0)
    t = torch.randn(4, 5, generator=g)
    indices, bits = [0, 7, 19], [30, 0, 31]
    out = FaultInjector(seed=0).flip_bits(t, indices, bits)
    diff = _xor_bits(t, out)
    expected = np.zeros(t.numel(), dtype=np.uint32)
    for i, b in zip(indices, bits):
        expected[i] = np.uint32(1 << b)
    assert np.array_equal(diff, expected)
    assert out.shape == t.shape and out.dtype == torch.float32


def test_input_tensor_is_not_modified() -> None:
    t = torch.arange(8, dtype=torch.float32)
    before = t.clone()
    FaultInjector().flip_bits(t, [3], [30])
    assert torch.equal(t, before)


def test_broadcast_single_bit_and_double_flip_cancels() -> None:
    t = torch.ones(6)
    inj = FaultInjector()
    out = inj.flip_bits(t, [1, 2], [23])
    assert int((_xor_bits(t, out) != 0).sum()) == 2
    assert torch.equal(inj.flip_bits(t, [4, 4], [5, 5]), t)


@pytest.mark.parametrize(
    "indices,bits",
    [([6], [0]), ([-1], [0]), ([0], [32]), ([0, 1], [1, 2, 3]), ([], [1])],
)
def test_invalid_requests_raise(indices: list[int], bits: list[int]) -> None:
    with pytest.raises(ValueError):
        FaultInjector().flip_bits(torch.zeros(6), indices, bits)


def test_non_float32_rejected() -> None:
    with pytest.raises(ValueError):
        FaultInjector().flip_bits(torch.zeros(4, dtype=torch.float64), [0], [1])


def test_inject_into_model_changes_only_target_and_logs() -> None:
    torch.manual_seed(0)
    model = TinyAttentionNet()
    before = {k: v.detach().clone() for k, v in model.named_tensors().items()}
    inj = FaultInjector(seed=0)
    records = inj.inject_into_model(model, "head.weight", [3], [30])
    assert len(records) == 1 and inj.log == records
    for name, tensor in model.named_tensors().items():
        changed = int((_xor_bits(before[name], tensor) != 0).sum())
        assert changed == (1 if name == "head.weight" else 0)
    assert records[0].old_bits != records[0].new_bits
    with pytest.raises(KeyError):
        inj.inject_into_model(model, "does.not.exist", [0])


def test_inject_random_is_deterministic_and_respects_exclusions() -> None:
    torch.manual_seed(0)
    base = TinyAttentionNet()
    runs = []
    for _ in range(2):
        m = copy.deepcopy(base)
        runs.append(FaultInjector(seed=7).inject_random(m, 5, exclude=[("head.weight", 0)]))
    assert runs[0] == runs[1] and len(runs[0]) == 5
    assert all(not (r.layer_name == "head.weight" and r.flat_index == 0) for r in runs[0])
