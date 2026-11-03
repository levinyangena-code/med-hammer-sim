# med-hammer-sim

> **SIMULATION ONLY.** This repository is a defensive red-team proof of concept. It applies
> bit flips to **in-memory PyTorch tensors**. It contains **no** Rowhammer or other hardware
> exploit code, **no** hardware or physical-memory access, **no** malware, **no** real patient
> data, and makes **no** network calls.

## What it does

`med-hammer-sim` studies one question from the AI-security literature: *can a neural Trojan
that is invisible during normal operation be activated by a hardware memory fault?* It models
this on a tiny attention-based classifier over synthetic, medical-image-like tensors
(a blurry bright "lesion" on a noisy background; binary label).

1. Train a clean classifier on synthetic data.
2. Implant a Trojan in a spare channel: a matched filter fires on a small corner trigger, but
   its path to the target logit goes through one **tiny dormant weight**.
3. Simulate a single exponent-MSB flip (bit 30) of that weight in float32. The weight grows by
   about 2^128, so **trigger + fault** forces the target label.
4. Measure clean accuracy, attack success rate (ASR), false-positive rate, and how well two
   baseline detectors (tensor checksums, activation-outlier scoring) catch the fault.

| Condition                | Expected behaviour                          |
|--------------------------|---------------------------------------------|
| clean input, no fault    | unchanged accuracy                          |
| clean input, fault       | unchanged (the dormant path sees zero input)|
| trigger, no fault        | Trojan dormant (ASR near 0)                 |
| trigger **and** fault    | forced to target label (ASR near 1)         |

## Threat model

* **Asset:** an AI model used in a medical-imaging pipeline, held in ordinary memory.
* **Attacker (assumed):** has already supplied a Trojaned model (e.g. via a poisoned supply
  chain) and can later cause a *single targeted bit flip* in the deployed weights, plus present
  inputs that carry the trigger. Real-world feasibility of that second capability depends on
  hardware, memory layout and mitigations, and is **outside the scope of this repo**.
* **Defender (assumed):** can checksum weights at load/run time and monitor activations on
  live traffic.
* **Why a simulation:** it lets defenders reason about detection coverage and mitigations
  without any exploit capability.

## Safe-simulation scope

* Faults are XORs on copies of `float32` tensors (`FaultInjector.flip_bits`), nothing more.
* Data is generated procedurally (`make_synthetic_dataset`); nothing is downloaded.
* Every run is seeded and deterministic on CPU.
* Outputs carry a `notice` field repeating the simulation-only statement.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

python -m medhammer run    --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1
python -m medhammer detect --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1
```

Without installing, use `PYTHONPATH=src python -m medhammer run`. Requires Python 3.11+.

### CLI flags

| Flag             | Meaning                                                                    |
|------------------|----------------------------------------------------------------------------|
| `--seed`         | master seed (data, init, training, fault sampling)                         |
| `--fault-rate`   | fraction of model elements that get an extra *collateral* random bit flip  |
| `--trigger-size` | trigger side length in pixels (2-4, must fit in one 4x4 patch)             |
| `--target-label` | class forced by the Trojan (0 or 1)                                        |
| `--epochs`       | training epochs (default 25)                                               |
| `--output`       | output path (`results.json` for `run`, `detection.json` for `detect`)      |

`run` writes `results.json` with: `config`, `metrics` (clean accuracy for baseline / Trojaned /
faulted models, `asr_trigger_only`, `asr_trigger_and_fault`, clean false-positive rates),
`fault_log` (every flip with old/new bit patterns), `detection` (precision/recall of the
checksum, activation and combined detectors on random synthetic faults) and
`targeted_fault_detection` (what the detectors say about the deliberate Trojan flip).

A typical default run shows clean accuracy unchanged by the implant, ASR 0.0 with the trigger
alone and 1.0 with trigger + fault. Checksums catch every injected fault (precision and recall
1.0), while activation scoring on clean traffic has high precision but low recall.

## Layout

```
src/medhammer/
  config.py          pydantic ExperimentConfig
  data.py            synthetic images + seeding
  model.py           TinyAttentionNet, named_tensors(), train_model()
  trojan.py          TrojanSpec, make_trigger_spec, apply_trigger, implant_trojan
  fault_injector.py  FaultInjector (flip_bits, inject_into_model, inject_random)
  eval.py            accuracy / ASR / false-positive rate, run_experiment()
  detection.py       checksums, activation-outlier scoring, precision/recall
  cli.py             typer app: run, detect
tests/               deterministic pytest suite
```

Source files contain `TODO(copilot)` comments that describe natural next features, so GitHub
Copilot can autocomplete them.

## Limitations

* Toy model, 16x16 synthetic images, two classes: results do **not** transfer to real
  medical models, real data, or real deployments.
* The Trojan is hand-constructed (matched filter + dormant weight), not learned or hidden by
  an optimizer; real Trojans are subtler.
* The trigger is a fixed corner pattern, and the clean model is trained to ignore it. A real
  attacker would need an equivalent step.
* The fault model is a perfectly targeted single bit flip. Real faults are probabilistic,
  layout-dependent and often blocked by ECC, TRR or other mitigations; `--fault-rate` only
  adds crude collateral noise.
* Only `float32` is supported; quantized and half-precision formats are not modelled.
* Baseline detectors are intentionally simple. Checksums only help if verified at the right
  time and from trusted storage. Activation scoring on clean traffic is blind to the targeted
  flip (the dormant path sees zero input); it only fires once triggered inputs arrive.
* Nothing here measures the real-world likelihood of a hardware fault attack.

## Extending toward real hardware (authorized labs only)

Do not do this on production systems, clinical systems, or anyone else's hardware.

* Keep the simulation as the reference implementation. Hide fault generation behind a small
  interface (e.g. a `FaultSource` protocol with `apply(model, plan)`), with the current
  `FaultInjector` as one implementation.
* Any hardware-backed implementation should live in a **separate, access-controlled
  repository** and run only on isolated lab equipment you own or are explicitly authorized
  to test, with written approval from your institution's security and ethics/compliance
  owners.
* Use only synthetic or properly de-identified, approved data. Never connect lab equipment to
  clinical networks, and follow responsible-disclosure practice for anything you find.
* Validate every result against the simulated expectations above, and report defensive
  findings (detection coverage, mitigations) rather than operational attack recipes.
