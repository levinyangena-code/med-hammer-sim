"""Software-only bit-flip simulation on in-memory float32 tensors.

This module never touches hardware, physical memory, or the network. It XORs bits in
copies of PyTorch tensors, which is the standard way to model memory faults in
DNN-robustness research.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from medhammer.model import TinyAttentionNet


class FlipRecord(BaseModel):
    """Audit record of one simulated bit flip.

    Attributes:
        layer_name: Parameter name.
        flat_index: Flat element index.
        bit_position: Bit flipped (0 = LSB, 31 = sign bit).
        old_value: Value before the flip.
        new_value: Value after the flip.
        old_bits: Hex bit pattern before the flip.
        new_bits: Hex bit pattern after the flip.
    """

    model_config = ConfigDict(frozen=True)

    layer_name: str
    flat_index: int
    bit_position: int
    old_value: float
    new_value: float
    old_bits: str
    new_bits: str


def _expand_pairs(
    indices: Sequence[int], bit_positions: Sequence[int], numel: int
) -> list[tuple[int, int]]:
    """Pair indices with bit positions (length-1 bit list is broadcast).

    Args:
        indices: Flat element indices.
        bit_positions: Bit positions; length 1 or ``len(indices)``.
        numel: Number of elements in the target tensor.

    Returns:
        List of ``(index, bit)`` pairs.

    Raises:
        ValueError: On empty input, length mismatch, or out-of-range values.
    """
    idx = [int(i) for i in indices]
    bits = [int(b) for b in bit_positions]
    if not idx or not bits:
        raise ValueError("indices and bit_positions must be non-empty")
    if len(bits) == 1:
        bits = bits * len(idx)
    if len(bits) != len(idx):
        raise ValueError("bit_positions must have length 1 or len(indices)")
    for i in idx:
        if not 0 <= i < numel:
            raise ValueError(f"index {i} out of range for tensor with {numel} elements")
    for b in bits:
        if not 0 <= b <= 31:
            raise ValueError(f"bit position {b} must be in [0, 31]")
    return list(zip(idx, bits))


class FaultInjector:
    """Simulated, controlled bit-flip injector.

    Args:
        seed: Seed for random fault sampling.
        default_bit: Bit used by :meth:`inject_into_model` when none is given.

    Attributes:
        log: Chronological list of :class:`FlipRecord` for every flip applied to a model.
    """

    def __init__(self, seed: int = 0, default_bit: int = 30) -> None:
        self._rng = np.random.default_rng(seed)
        self.default_bit = default_bit
        self.log: list[FlipRecord] = []

    def flip_bits(
        self,
        tensor: torch.Tensor,
        indices: Sequence[int],
        bit_positions: Sequence[int],
    ) -> torch.Tensor:
        """XOR selected bits of selected elements and return a new tensor.

        Pairs are applied in order, so flipping the same bit of the same element twice
        cancels out. The input tensor is not modified.

        Args:
            tensor: float32 tensor.
            indices: Flat element indices (row-major).
            bit_positions: Bit positions (0 = LSB, 31 = sign). Either one value
                (applied to every index) or one value per index.

        Returns:
            A new float32 tensor, same shape and device as ``tensor``.

        Raises:
            ValueError: If dtype is not float32, or indices/bits are invalid.
        """
        if tensor.dtype is not torch.float32:
            raise ValueError("flip_bits supports float32 tensors only")
        pairs = _expand_pairs(indices, bit_positions, tensor.numel())
        flat = tensor.detach().cpu().contiguous().numpy().copy().reshape(-1)
        raw = flat.view(np.uint32)
        for i, b in pairs:
            raw[i] ^= np.uint32(1 << b)
        return torch.from_numpy(flat.reshape(tuple(tensor.shape))).to(tensor.device)

    @torch.no_grad()
    def inject_into_model(
        self,
        model: TinyAttentionNet,
        layer_name: str,
        indices: Sequence[int],
        bit_positions: Sequence[int] | None = None,
    ) -> list[FlipRecord]:
        """Flip bits in one named parameter of ``model`` in place.

        Args:
            model: Model whose ``named_tensors()`` contains ``layer_name``.
            layer_name: Parameter name, e.g. ``"head.weight"``.
            indices: Flat element indices.
            bit_positions: Bits to flip; defaults to ``[self.default_bit]``.

        Returns:
            One :class:`FlipRecord` per flip (also appended to ``self.log``).

        Raises:
            KeyError: If ``layer_name`` is not a model tensor.
            ValueError: If indices or bit positions are invalid.
        """
        tensors = model.named_tensors()
        if layer_name not in tensors:
            raise KeyError(f"unknown tensor '{layer_name}'; options: {sorted(tensors)}")
        param = tensors[layer_name]
        bits = list(bit_positions) if bit_positions is not None else [self.default_bit]
        pairs = _expand_pairs(indices, bits, param.numel())
        records: list[FlipRecord] = []
        for i, b in pairs:
            before = param.detach().reshape(-1)[i : i + 1].clone()
            param.copy_(self.flip_bits(param.detach(), [i], [b]))
            after = param.detach().reshape(-1)[i : i + 1].clone()
            records.append(
                FlipRecord(
                    layer_name=layer_name,
                    flat_index=i,
                    bit_position=b,
                    old_value=float(before),
                    new_value=float(after),
                    old_bits=f"0x{int(before.numpy().view(np.uint32)[0]):08x}",
                    new_bits=f"0x{int(after.numpy().view(np.uint32)[0]):08x}",
                )
            )
        self.log.extend(records)
        return records

    def inject_random(
        self,
        model: TinyAttentionNet,
        n_flips: int,
        bit_range: tuple[int, int] = (0, 32),
        exclude: Iterable[tuple[str, int]] = (),
    ) -> list[FlipRecord]:
        """Flip ``n_flips`` random bits at distinct random elements across the model.

        Args:
            model: Model to fault in place.
            n_flips: Number of distinct elements to fault.
            bit_range: Half-open range ``[lo, hi)`` of bit positions to sample from.
            exclude: ``(layer_name, flat_index)`` sites that must not be chosen.

        Returns:
            The :class:`FlipRecord` list.

        Raises:
            ValueError: If ``n_flips`` is negative/too large or ``bit_range`` is invalid.
        """
        lo, hi = bit_range
        if not 0 <= lo < hi <= 32:
            raise ValueError("bit_range must satisfy 0 <= lo < hi <= 32")
        if n_flips < 0:
            raise ValueError("n_flips must be >= 0")
        tensors = model.named_tensors()
        names = list(tensors)
        offsets = np.concatenate([[0], np.cumsum([tensors[n].numel() for n in names])])
        total = int(offsets[-1])
        banned = {int(offsets[names.index(n)]) + int(i) for n, i in exclude}
        allowed = np.setdiff1d(np.arange(total), np.array(sorted(banned), dtype=np.int64))
        if n_flips > len(allowed):
            raise ValueError("n_flips exceeds number of eligible elements")
        if n_flips == 0:
            return []
        chosen = self._rng.choice(allowed, size=n_flips, replace=False)
        bits = self._rng.integers(lo, hi, size=n_flips)
        records: list[FlipRecord] = []
        for g, b in zip(chosen, bits):
            t = int(np.searchsorted(offsets, g, side="right") - 1)
            records += self.inject_into_model(model, names[t], [int(g - offsets[t])], [int(b)])
        return records


# TODO(copilot): support float16 / bfloat16 / int8-quantized tensors in `flip_bits`
#   (use the matching unsigned numpy view and bit width; validate bit positions per dtype).
# TODO(copilot): add `restore(model, records)` that undoes a list of FlipRecord entries.
