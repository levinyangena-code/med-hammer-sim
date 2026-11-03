"""Simulated neural Trojan that stays dormant until a bit flip activates it.

Mechanism (toy, for simulation only):
  * A matched-filter unit in the spare channel fires only on the trigger pattern.
  * Its path to the target-class logit goes through one tiny "dormant" weight, so the
    Trojan has negligible effect on outputs.
  * Flipping the exponent MSB of that weight (in simulation) amplifies it by ~2^128,
    so trigger + fault forces the target label. Trigger alone or fault alone does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from medhammer.model import TinyAttentionNet


@dataclass(frozen=True, eq=False)
class TrojanSpec:
    """Description of a Trojan.

    Attributes:
        trigger_pattern: ``(H, W)`` tensor of trigger pixel values (used where mask=1).
        trigger_mask: ``(H, W)`` tensor of {0, 1} marking trigger pixels.
        target_label: Class forced when the Trojan is active.
        dormant_weight: Tiny weight keeping the Trojan silent before a fault.
        fault_bit: Bit position to flip at the Trojan site (30 = exponent MSB).

    Raises:
        ValueError: If shapes, label, weight or bit are invalid.
    """

    trigger_pattern: torch.Tensor
    trigger_mask: torch.Tensor
    target_label: int
    dormant_weight: float = 1e-4
    fault_bit: int = 30

    def __post_init__(self) -> None:
        if self.trigger_pattern.shape != self.trigger_mask.shape or self.trigger_mask.ndim != 2:
            raise ValueError("trigger_pattern and trigger_mask must be 2-D with equal shape")
        if self.target_label < 0:
            raise ValueError("target_label must be >= 0")
        if not 0.0 < abs(self.dormant_weight) < 1.0:
            raise ValueError("dormant_weight must satisfy 0 < |w| < 1")
        if not 0 <= self.fault_bit <= 31:
            raise ValueError("fault_bit must be in [0, 31]")
        if float(self.trigger_mask.sum()) < 1:
            raise ValueError("trigger_mask must be non-empty")

    @property
    def trigger_size(self) -> int:
        """Side length of the (square) trigger.

        Returns:
            Integer side length.
        """
        return int(round(float(self.trigger_mask.sum()) ** 0.5))


@dataclass(frozen=True)
class FaultSite:
    """Location of the bit whose flip activates the Trojan.

    Attributes:
        layer_name: Name of the parameter tensor.
        index: Flat element index in that tensor.
        bit_position: Bit to flip.
    """

    layer_name: str
    index: int
    bit_position: int


def make_trigger_spec(
    image_size: int,
    trigger_size: int,
    target_label: int,
    dormant_weight: float = 1e-4,
    fault_bit: int = 30,
) -> TrojanSpec:
    """Build a checkerboard trigger in the bottom-right corner.

    Args:
        image_size: Image height/width.
        trigger_size: Trigger side length (>= 2).
        target_label: Class forced by the Trojan.
        dormant_weight: Dormant head weight.
        fault_bit: Bit to flip at the Trojan site.

    Returns:
        A :class:`TrojanSpec`.

    Raises:
        ValueError: If the trigger size is out of range.
    """
    if not 2 <= trigger_size <= image_size:
        raise ValueError("trigger_size must be in [2, image_size]")
    mask = torch.zeros(image_size, image_size)
    pattern = torch.zeros(image_size, image_size)
    for i in range(trigger_size):
        for j in range(trigger_size):
            r, c = image_size - trigger_size + i, image_size - trigger_size + j
            mask[r, c] = 1.0
            pattern[r, c] = 1.0 if (i + j) % 2 == 0 else 0.0
    return TrojanSpec(pattern, mask, target_label, dormant_weight, fault_bit)


def apply_trigger(batch: torch.Tensor, spec: TrojanSpec) -> torch.Tensor:
    """Stamp the trigger onto a batch of images.

    Args:
        batch: Images of shape ``(B, C, H, W)``.
        spec: Trojan specification.

    Returns:
        A new tensor with trigger pixels overwritten.

    Raises:
        ValueError: If the spatial size does not match the trigger mask.
    """
    if batch.shape[-2:] != spec.trigger_mask.shape:
        raise ValueError("batch spatial size does not match trigger mask")
    mask = spec.trigger_mask.to(batch.dtype)
    return batch * (1.0 - mask) + spec.trigger_pattern.to(batch.dtype) * mask


@torch.no_grad()
def implant_trojan(model: TinyAttentionNet, spec: TrojanSpec) -> FaultSite:
    """Implant the dormant Trojan into ``model`` in place.

    Args:
        model: Model to modify.
        spec: Trojan specification.

    Returns:
        The :class:`FaultSite` whose bit flip activates the Trojan.

    Raises:
        ValueError: If the label is out of range or the trigger does not lie
            entirely inside the bottom-right patch.
    """
    p, s = model.patch_size, model.spare_channel
    if spec.target_label >= model.num_classes:
        raise ValueError("target_label out of range for this model")
    if spec.trigger_mask.shape != (model.image_size, model.image_size):
        raise ValueError("trigger mask size does not match model image size")
    rows, cols = torch.nonzero(spec.trigger_mask, as_tuple=True)
    if int(rows.min()) < model.image_size - p or int(cols.min()) < model.image_size - p:
        raise ValueError("trigger must fit inside the bottom-right patch")

    model.zero_spare_channel()
    full_kernel = 2.0 * (spec.trigger_pattern - 0.5) * spec.trigger_mask  # +1 / -1 matched filter
    response = float((full_kernel * spec.trigger_pattern).sum())
    model.embed.weight[s, 0] = full_kernel[-p:, -p:]
    model.embed.bias[s] = -0.75 * response  # clean inputs in [0, 1] stay below threshold
    model.head.weight[spec.target_label, s] = spec.dormant_weight
    index = spec.target_label * model.head.weight.shape[1] + s
    return FaultSite("head.weight", index, spec.fault_bit)


# TODO(copilot): add `make_trigger_spec(..., corner: Literal["br", "bl", "tr", "tl"])` and
#   generalize `implant_trojan` to the patch that contains the chosen corner.
# TODO(copilot): add `remove_trojan(model, site)` that restores the spare channel to zero.
