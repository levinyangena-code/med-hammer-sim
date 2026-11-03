"""Command-line interface: ``python -m medhammer run|detect``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from medhammer.config import SIMULATION_NOTICE, ExperimentConfig
from medhammer.eval import prepare_experiment, run_detection, run_experiment

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="med-hammer-sim: software-only fault-induced Trojan simulation. " + SIMULATION_NOTICE,
)

Seed = Annotated[int, typer.Option(help="Master random seed.")]
FaultRate = Annotated[
    float, typer.Option(min=0.0, max=1.0, help="Fraction of model elements given collateral flips.")
]
TriggerSize = Annotated[int, typer.Option(min=2, max=4, help="Trigger side length in pixels.")]
TargetLabel = Annotated[int, typer.Option(min=0, max=1, help="Label the Trojan forces.")]
Epochs = Annotated[int, typer.Option(min=1, help="Training epochs for the clean model.")]


@app.command()
def run(
    seed: Seed = 0,
    fault_rate: FaultRate = 0.001,
    trigger_size: TriggerSize = 3,
    target_label: TargetLabel = 1,
    epochs: Epochs = 25,
    output: Annotated[Path, typer.Option(help="Where to write the JSON results.")] = Path(
        "results.json"
    ),
) -> None:
    """Run the full experiment (attack metrics + detection) and write results.json."""
    cfg = ExperimentConfig(
        seed=seed,
        fault_rate=fault_rate,
        trigger_size=trigger_size,
        target_label=target_label,
        epochs=epochs,
    )
    result = run_experiment(cfg)
    output.write_text(result.model_dump_json(indent=2))
    m = result.metrics
    typer.echo(SIMULATION_NOTICE)
    typer.echo(
        f"clean acc {m.clean_accuracy_baseline:.3f} -> {m.clean_accuracy_trojaned:.3f} (trojaned) | "
        f"ASR trigger-only {m.asr_trigger_only:.3f} | ASR trigger+fault {m.asr_trigger_and_fault:.3f}"
    )
    typer.echo(f"wrote {output}")


@app.command()
def detect(
    seed: Seed = 0,
    fault_rate: FaultRate = 0.001,
    trigger_size: TriggerSize = 3,
    target_label: TargetLabel = 1,
    epochs: Epochs = 25,
    output: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path(
        "detection.json"
    ),
) -> None:
    """Evaluate the baseline detectors on synthetic faults and write detection.json."""
    cfg = ExperimentConfig(
        seed=seed,
        fault_rate=fault_rate,
        trigger_size=trigger_size,
        target_label=target_label,
        epochs=epochs,
    )
    report, targeted = run_detection(prepare_experiment(cfg))
    output.write_text(
        '{"detection": ' + report.model_dump_json() + ', "targeted_fault_detection": '
        + targeted.model_dump_json() + "}"
    )
    typer.echo(SIMULATION_NOTICE)
    typer.echo(
        f"checksum P/R {report.checksum.precision:.2f}/{report.checksum.recall:.2f} | "
        f"activation P/R {report.activation.precision:.2f}/{report.activation.recall:.2f}"
    )
    typer.echo(f"wrote {output}")


# TODO(copilot): add a `sweep` command: --rates 0,0.001,0.01 --output sweep.json that calls
#   `sweep_fault_rate` and writes one JSON list of AttackMetrics.
