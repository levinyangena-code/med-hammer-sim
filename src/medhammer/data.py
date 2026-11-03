"""Synthetic, PHI-free, medical-image-like data. Nothing is downloaded."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for deterministic runs.

    Args:
        seed: Seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_dataset(
    n: int, image_size: int = 16, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic grayscale "scans" with a binary label.

    Label 1 images contain a bright Gaussian "lesion" blob on a noisy background;
    label 0 images contain only background texture. Pixel values lie in [0, 1].

    Args:
        n: Number of images.
        image_size: Height and width in pixels.
        seed: Seed for the local random generator.

    Returns:
        A tuple ``(images, labels)`` with shapes ``(n, 1, H, W)`` (float32) and
        ``(n,)`` (int64).

    Raises:
        ValueError: If ``n`` or ``image_size`` is too small.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if image_size < 8:
        raise ValueError("image_size must be >= 8")
    g = torch.Generator().manual_seed(seed)
    labels = (torch.arange(n) % 2)[torch.randperm(n, generator=g)].to(torch.int64)

    background = 0.3 * torch.rand(n, image_size, image_size, generator=g)
    ys = torch.arange(image_size, dtype=torch.float32).view(1, -1, 1)
    xs = torch.arange(image_size, dtype=torch.float32).view(1, 1, -1)
    cy = 3 + (image_size - 6) * torch.rand(n, 1, 1, generator=g)
    cx = 3 + (image_size - 6) * torch.rand(n, 1, 1, generator=g)
    sigma = 1.5 + torch.rand(n, 1, 1, generator=g)
    amp = 0.5 + 0.3 * torch.rand(n, 1, 1, generator=g)
    blob = amp * torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2))

    images = background + blob * labels.view(-1, 1, 1).float()
    return images.clamp(0.0, 1.0).unsqueeze(1).contiguous(), labels


# TODO(copilot): add `make_multiclass_dataset(n, image_size, num_classes, seed)` returning
#   synthetic images with one distinct blob shape per class (still no real data, no downloads).
# TODO(copilot): add an optional `noise_level: float` argument to `make_synthetic_dataset`.
