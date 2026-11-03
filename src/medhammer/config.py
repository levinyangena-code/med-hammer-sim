"""Experiment configuration (validated with pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SIMULATION_NOTICE = (
    "SIMULATION ONLY: bit flips are applied to in-memory PyTorch tensors. "
    "No hardware access, no exploit code, no real patient data, no network calls."
)


class ExperimentConfig(BaseModel):
    """Immutable configuration for one simulated red-team experiment.

    Attributes:
        seed: Master seed for data, model init, training and fault sampling.
        fault_rate: Fraction of model tensor elements that receive an additional
            *collateral* random bit flip (models an imprecise fault mechanism).
        trigger_size: Side length (pixels) of the square trigger in the image corner.
        target_label: Class the Trojan forces when it activates.
        image_size: Height/width of the synthetic square images.
        patch_size: Patch size of the convolutional patch embedding.
        embed_dim: Number of functional embedding channels (one spare channel is added).
        n_train: Number of synthetic training images.
        n_test: Number of synthetic test images.
        epochs: Training epochs for the clean model.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        dormant_weight: Tiny head weight that keeps the Trojan silent until faulted.
        fault_bit: Bit position (0-31) flipped at the Trojan site (30 = exponent MSB).
        detection_trials: Number of synthetic-fault trials for detector evaluation.
        detection_threshold: Activation-outlier z-score threshold.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    fault_rate: float = Field(0.001, ge=0.0, le=1.0)
    trigger_size: int = Field(3, ge=2, le=4)
    target_label: int = Field(1, ge=0, le=1)
    image_size: int = Field(16, ge=8)
    patch_size: int = Field(4, ge=2)
    embed_dim: int = Field(16, ge=4)
    n_train: int = Field(512, ge=16)
    n_test: int = Field(256, ge=16)
    epochs: int = Field(25, ge=1)
    lr: float = Field(1e-2, gt=0.0)
    batch_size: int = Field(64, ge=1)
    dormant_weight: float = Field(1e-4, gt=0.0, lt=1.0)
    fault_bit: int = Field(30, ge=0, le=31)
    detection_trials: int = Field(40, ge=2)
    detection_threshold: float = Field(10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_geometry(self) -> "ExperimentConfig":
        """Validate cross-field geometry constraints.

        Returns:
            The validated config.

        Raises:
            ValueError: If the image is not divisible by the patch size, or the
                trigger does not fit inside a single patch.
        """
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.trigger_size > self.patch_size:
            raise ValueError("trigger_size must be <= patch_size")
        return self
