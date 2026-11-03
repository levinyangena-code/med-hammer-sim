"""Small attention-based classifier whose parameters can be targeted by name."""

from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """Single-head self-attention over patch tokens.

    Args:
        dim: Token width.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            tokens: Tensor of shape ``(B, N, D)``.

        Returns:
            Tensor of shape ``(B, N, D)``.
        """
        scores = self.q(tokens) @ self.k(tokens).transpose(-2, -1) / math.sqrt(self.dim)
        return self.proj(torch.softmax(scores, dim=-1) @ self.v(tokens))


class TinyAttentionNet(nn.Module):
    """Patch-embedding + self-attention + linear head.

    The token width is ``embed_dim + 1``: the last channel is a *spare* channel that
    is silent (all-zero weights) in the clean model. It models unused capacity that a
    Trojan can hide in.

    Args:
        image_size: Input height/width.
        patch_size: Patch size for the strided convolution embedding.
        embed_dim: Number of functional channels (spare channel is added on top).
        num_classes: Number of output classes.

    Raises:
        ValueError: If ``image_size`` is not divisible by ``patch_size``.
    """

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        embed_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.width = embed_dim + 1
        self.spare_channel = embed_dim
        self.embed = nn.Conv2d(1, self.width, kernel_size=patch_size, stride=patch_size)
        self.attn = SelfAttention(self.width)
        self.head = nn.Linear(self.width, num_classes)
        self.zero_spare_channel()

    @torch.no_grad()
    def zero_spare_channel(self) -> None:
        """Zero every weight that reads from or writes to the spare channel.

        Returns:
            None.
        """
        s = self.spare_channel
        self.embed.weight[s].zero_()
        self.embed.bias[s].zero_()
        for lin in (self.attn.q, self.attn.k, self.attn.v, self.attn.proj):
            lin.weight[s, :].zero_()
            lin.weight[:, s].zero_()
            lin.bias[s].zero_()
        self.head.weight[:, s].zero_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pooled token features.

        Args:
            x: Images of shape ``(B, 1, H, W)``.

        Returns:
            Tensor of shape ``(B, width)``.
        """
        tokens = torch.relu(self.embed(x)).flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(tokens)
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Images of shape ``(B, 1, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        return self.head(self.forward_features(x))

    def named_tensors(self) -> dict[str, torch.Tensor]:
        """Expose all parameters by name so faults can target them.

        Returns:
            Ordered mapping from parameter name to the live parameter tensor.
        """
        return dict(self.named_parameters())


def train_model(
    model: TinyAttentionNet,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 25,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = 0,
) -> list[float]:
    """Train the model with Adam and cross-entropy.

    Args:
        model: Model to train in place.
        x: Training images ``(N, 1, H, W)``.
        y: Integer labels ``(N,)``.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        seed: Seed for batch shuffling.

    Returns:
        Mean training loss per epoch.

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            model.zero_spare_channel()  # keep spare capacity silent while training
            total += float(loss.detach()) * len(idx)
        history.append(total / len(x))
    model.eval()
    return history


# TODO(copilot): add an `n_heads` option to SelfAttention (keep `named_tensors()` names stable).
# TODO(copilot): add `save_checkpoint(model, path)` / `load_checkpoint(path)` helpers
#   that also store the reference checksums from `medhammer.detection.model_checksums`.
