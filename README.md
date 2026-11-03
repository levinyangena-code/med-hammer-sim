## `README.md`

````markdown
<div align="center">

# med-hammer-sim

**A software-only simulation of fault-induced neural Trojan activation in a toy medical-imaging classifier.**

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch, CPU" src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Scope: simulation only" src="https://img.shields.io/badge/scope-simulation%20only-orange">
  <img alt="Data: synthetic only" src="https://img.shields.io/badge/data-synthetic%20only-blue">
  <img alt="License: not set yet" src="https://img.shields.io/badge/license-not%20set-lightgrey">
</p>

</div>

> **Simulation only.** This repository applies bit flips to **in-memory PyTorch tensors**. It contains no
> Rowhammer or other hardware-exploit code, no hardware or physical-memory access, no malware, no real
> patient data, and no network code. Results from the toy model do not transfer to real systems.

<p align="center">
  <img src="docs/med-hammer-sim-flow.svg" width="100%"
       alt="Architecture diagram: inputs (synthetic images, config and CLI flags, trigger spec) feed a simulation core (model with dormant Trojan, simulated bit flips, metrics runner), then checks (tensor checksums, activation z-score, scope guardrail), producing results.json, detection.json and console output. Bars show one sample run: ASR 0.00 with trigger only, 1.00 with trigger plus fault, checksum recall 1.00, activation recall 0.15.">
</p>

## Overview

`med-hammer-sim` explores one defensive question: *can a neural Trojan that is invisible in normal operation be
switched on by a single memory bit flip?* It builds a tiny attention-based classifier over synthetic,
medical-image-like tensors, hides a dormant Trojan in it, simulates the bit flip in software, and measures what
changes and whether two simple detectors notice.

It is meant for security researchers and ML-safety engineers who want a small, deterministic, fully inspectable
testbed for reasoning about detection coverage. It is not a benchmark and not an attack toolkit.

## How it works

1. **Data.** `make_synthetic_dataset` draws noisy 16x16 grayscale images; label 1 images contain a bright Gaussian
   "lesion" blob. Nothing is downloaded.
2. **Clean model.** `TinyAttentionNet` (patch embedding, single-head self-attention, linear head) is trained with
   Adam. The trigger is stamped on half of the training images *with their true labels*, so the clean pathway
   learns to ignore it.
3. **Trojan.** The model has one spare channel that is silent in the clean model. `implant_trojan` turns it into a
   matched filter for a small checkerboard trigger in the bottom-right corner. Its only route to the target-class
   logit is one **dormant weight** (`1e-4`), so the Trojan has almost no effect on outputs.
4. **Fault.** `FaultInjector` XORs bit 30 (the float32 exponent MSB) of that weight in memory. In the sample run the
   recorded value goes from `1e-4` to about `3.4e34`, so trigger **and** fault force the target label.
5. **Measure.** `eval.py` reports accuracy, attack success rate (ASR) and false-positive rate; `detection.py` scores
   tensor checksums and activation outliers.

```mermaid
flowchart LR
    A["Synthetic images<br/>data.py"] --> C["Train clean model<br/>model.py"]
    B["Config + CLI flags<br/>config.py, cli.py"] --> C
    C --> D["Implant dormant Trojan<br/>trojan.py"]
    D --> E["Simulated bit flip<br/>fault_injector.py"]
    E --> F["Accuracy, ASR, FPR<br/>eval.py"]
    D --> G["Reference checksums<br/>detection.py"]
    E --> H["Checksum + activation detectors<br/>detection.py"]
    G --> H
    F -->|run| I[("results.json")]
    H -->|run| I
    H -->|detect| J[("detection.json")]
```

| Condition               | Behaviour verified by the tests / sample run                        |
|-------------------------|---------------------------------------------------------------------|
| clean input, no fault   | outputs match the un-implanted model                                |
| clean input, fault      | outputs match the un-implanted model (the dormant path sees zero)   |
| trigger, no fault       | Trojan dormant; outputs match the un-implanted model (within 1e-3) |
| trigger **and** fault   | every triggered sample is classified as the target label            |

## Quickstart

Requires Python 3.11+. PyTorch runs on CPU; no GPU is needed.

```bash
git clone <your-repo-url> med-hammer-sim    # replace with your repository URL
cd med-hammer-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest
python -m medhammer run    --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1
python -m medhammer detect --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1
```

Without installing, prefix commands with `PYTHONPATH=src`. After installing, the `medhammer` console script is
equivalent to `python -m medhammer`. Real output from the two commands above (seed 0):

<p align="center">
  <img src="docs/cli-demo.svg" width="100%"
       alt="Animated terminal showing python -m medhammer run printing clean accuracy 1.000 to 1.000, ASR trigger-only 0.000 and ASR trigger+fault 1.000, then python -m medhammer detect printing checksum precision/recall 1.00/1.00 and activation precision/recall 1.00/0.15.">
</p>

## CLI and Python API reference

Both commands take the same flags.

| Command | Flag             | Default        | Meaning                                                                  |
|---------|------------------|----------------|--------------------------------------------------------------------------|
| `run`, `detect` | `--seed`         | `0`            | Master seed for data, model init, training and fault sampling            |
| `run`, `detect` | `--fault-rate`   | `0.001` (0 to 1) | Fraction of model tensor elements to fault (see note below)            |
| `run`, `detect` | `--trigger-size` | `3` (2 to 4)   | Trigger side length in pixels; must fit in one 4x4 patch                 |
| `run`, `detect` | `--target-label` | `1` (0 or 1)   | Class the Trojan forces                                                  |
| `run`, `detect` | `--epochs`       | `25`           | Training epochs for the clean model                                      |
| `run`   | `--output`       | `results.json` | Where to write the full results                                          |
| `detect`| `--output`       | `detection.json` | Where to write the detector report                                     |

**`--fault-rate` note.** The model has 1,549 tensor elements. In `run`, the targeted Trojan flip is always applied
and `round(fault_rate x 1549)` extra random bit flips ("collateral" faults, any bit 0 to 31) are added. In `detect`,
each faulted trial flips `max(1, round(fault_rate x 1549))` random exponent bits (bits 23 to 30). At the default
`0.001` that is 2 flips in both commands.

| Python entry point | Module | Purpose |
|---|---|---|
| `make_synthetic_dataset(n, image_size, seed)` | `data.py` | Synthetic images and binary labels |
| `TinyAttentionNet`, `train_model(...)`, `.named_tensors()` | `model.py` | Model, training, access to parameters by name |
| `make_trigger_spec`, `apply_trigger`, `implant_trojan` | `trojan.py` | Build/stamp the trigger, implant the Trojan, get the `FaultSite` |
| `FaultInjector.flip_bits`, `.inject_into_model`, `.inject_random` | `fault_injector.py` | Simulated bit flips with an audit log of `FlipRecord`s |
| `accuracy`, `attack_success_rate`, `false_positive_rate` | `eval.py` | Metrics |
| `prepare_experiment`, `run_attack_evaluation`, `run_detection`, `run_experiment` | `eval.py` | Pieces of the experiment and the full runner |
| `model_checksums`, `fit_activation_baseline`, `activation_outlier_score`, `evaluate_detector`, `evaluate_targeted_fault` | `detection.py` | Baseline detectors and their evaluation |

## Metrics and outputs

`results.json` (from `run`) contains `notice`, `config`, `metrics`, `fault_log`, `detection` and
`targeted_fault_detection`. `detection.json` (from `detect`) contains only `detection` and
`targeted_fault_detection`; it has no `notice` field, but the CLI prints the notice to the console.

| Output | Definition |
|---|---|
| `clean_accuracy_baseline` / `_trojaned` / `_faulted` | Accuracy on the synthetic test set (256 images by default) for the clean, Trojaned and faulted models |
| `asr_trigger_only` | Among test images whose true label is not the target, fraction predicted as the target after stamping the trigger, on the Trojaned model with no fault |
| `asr_trigger_and_fault` | Same, on the faulted model (targeted flip plus collateral flips) |
| `fpr_clean_trojaned` / `_faulted` | Among clean (untriggered) non-target images, fraction predicted as the target label |
| `n_collateral_flips` | Number of extra random flips in `run` |
| `fault_log` | Every flip: layer, flat index, bit, old/new value and bit pattern |
| `detection.{checksum,activation,combined}` | tp / fp / tn / fn, precision, recall over random synthetic faults (precision and recall are 0.0 when undefined) |
| `targeted_fault_detection` | Whether checksums and activation scoring (on clean traffic and on triggered traffic) flag the deliberate Trojan flip |

**Sample values.** Produced by `python -m medhammer run --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1`
(all other settings at their defaults) with Python 3.12.3, torch 2.15.0 (CPU), numpy 2.4.4, pydantic 2.14.2 and
typer 0.27.3. Other versions or hardware may give different numbers. The bars and terminal in `docs/*.svg` are static
snapshots of this run and do not update automatically.

| Metric (seed 0)                          | Value |
|------------------------------------------|-------|
| clean accuracy: baseline / Trojaned / faulted | 1.000 / 1.000 / 1.000 |
| ASR, trigger only                        | 0.000 |
| ASR, trigger + fault                     | 1.000 |
| clean FPR: Trojaned / faulted            | 0.000 / 0.000 |
| collateral flips                         | 2 |

| Detector on random faults (40 trials, 20 faulted, 2 flips each, threshold 10) | tp | fp | tn | fn | precision | recall |
|---|---|---|---|---|---|---|
| checksum   | 20 | 0 | 20 | 0  | 1.00 | 1.00 |
| activation | 3  | 0 | 20 | 17 | 1.00 | 0.15 |
| combined (either)  | 20 | 0 | 20 | 0  | 1.00 | 1.00 |

For the deliberate Trojan flip: checksums detect it; the activation score on clean traffic is about 2.29 (below the
threshold of 10, so not flagged); on triggered traffic it hits the 1e12 cap (flagged). The synthetic task is easy,
so accuracy of 1.000 says nothing about clinical performance.

## Project layout

```
med-hammer-sim/
├── pyproject.toml
├── README.md
├── .gitignore
├── .github/copilot-instructions.md
├── docs/
│   ├── med-hammer-sim-flow.svg
│   └── cli-demo.svg
├── src/medhammer/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── trojan.py
│   ├── fault_injector.py
│   ├── eval.py
│   ├── detection.py
│   └── cli.py
└── tests/
    ├── test_fault_injector.py
    ├── test_trojan.py
    ├── test_eval.py
    └── test_detection.py
```

| File | Role |
|---|---|
| `config.py` | Frozen pydantic `ExperimentConfig` with validation (image size divisible by patch size; trigger fits in a patch) |
| `data.py` | Seeding helper and synthetic data generator |
| `model.py` | `TinyAttentionNet` (with a spare channel), `named_tensors()`, `train_model` |
| `trojan.py` | `TrojanSpec`, `FaultSite`, trigger construction and stamping, Trojan implant |
| `fault_injector.py` | float32 bit-flip simulation, `FlipRecord` audit records, random fault sampling |
| `eval.py` | Metrics, `PreparedExperiment`, `run_experiment` and the result schemas |
| `detection.py` | SHA-256 checksums, activation-outlier scoring, detector evaluation |
| `cli.py`, `__main__.py` | Typer app (`run`, `detect`) and the `python -m medhammer` entry point |
| `.github/copilot-instructions.md` | Project rules for GitHub Copilot, including the simulation-only constraints |

## Testing

```bash
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`. At the time of writing the suite has 23
tests and passes in a few seconds on CPU. All tests are seeded.

| File | Covers |
|---|---|
| `tests/test_fault_injector.py` | exactly the requested bits flip; input not mutated; invalid indices/bits/dtypes rejected; model injection touches one tensor and logs it; random injection is deterministic and honours exclusions |
| `tests/test_trojan.py` | behaviour for clean/trigger crossed with fault/no fault; spare channel silent on clean inputs; trigger changes only masked pixels; invalid specs rejected |
| `tests/test_eval.py` | clean accuracy unchanged by the implant; ASR needs trigger and fault; `run_experiment` is deterministic; the CLI writes `results.json` |
| `tests/test_detection.py` | checksum flags a flipped tensor; activation score flags a large fault; detector precision/recall and determinism; the targeted flip evades activation scoring on clean traffic but not checksums |

## Safety boundaries

- Faults are XORs on in-memory `float32` tensors: `flip_bits` returns a new tensor and `inject_into_model` edits the
  model you pass it. Nothing else is touched.
- Data is generated procedurally; the package reads no datasets and downloads nothing.
- The package contains no network code, no hardware or driver access, and no OS memory APIs.
- `results.json` carries a `notice` field and the CLI prints the same simulation-only statement.
- `.github/copilot-instructions.md` records these rules so that AI-assisted contributions keep them.

## Adapting and extending

- **Data:** replace `make_synthetic_dataset` with another procedural generator that returns images `(N, 1, H, W)` and
  integer labels `(N,)`. Do not wire in real patient data.
- **Model:** `implant_trojan` depends on `TinyAttentionNet`'s structure (`embed`, `head`, `spare_channel`,
  `zero_spare_channel`), so a different architecture needs a matching implant.
- **Detectors:** add a function in `detection.py` and include it in `evaluate_detector`, which currently scores
  the checksum and activation detectors.
- **Fault models:** call `FaultInjector.flip_bits`, `inject_into_model` or `inject_random` (`bit_range` selects the
  bits sampled) from your own experiment code.
- **Real hardware:** not provided and not supported here. If you ever extend toward hardware, do it only in an
  authorized, isolated lab on equipment you own or are explicitly cleared to test, with institutional approval,
  synthetic or approved de-identified data, and no connection to clinical networks. Keep any hardware-backed code in a
  separate, access-controlled repository.

## Limitations

- Toy model, 16x16 synthetic images, two classes; results do not transfer to real medical models or deployments.
- The Trojan is hand-constructed (matched filter plus dormant weight), not learned or concealed by an optimizer.
- The trigger is a fixed bottom-right checkerboard, and the clean model is trained to ignore it; a real attacker
  would need an equivalent step.
- The targeted fault is a perfectly placed single bit flip. Real faults are probabilistic and layout-dependent, and
  `--fault-rate` only adds crude random noise.
- Only `float32` tensors are supported.
- The detectors are deliberately simple. Checksums only help if verified from trusted storage at the right time.
  Activation scoring on clean traffic does not see the targeted flip, because the dormant path receives zero input.
- The sample numbers come from one seed on one software stack and are not a benchmark.
- Determinism is expected on CPU with the same library versions; it is not guaranteed across versions or hardware.

## Roadmap

Planned items (they exist only as `TODO(copilot)` comments in the source and are **not implemented**):

- [ ] Multi-class synthetic data and a noise-level option (`data.py`)
- [ ] Multi-head attention and checkpoint helpers that store reference checksums (`model.py`)
- [ ] Selectable trigger corner and a `remove_trojan` helper (`trojan.py`)
- [ ] float16 / bfloat16 / int8 support and a `restore` method (`fault_injector.py`)
- [ ] Weight-range detector and threshold sweeps (`detection.py`)
- [ ] Fault-rate sweeps and bootstrap confidence intervals (`eval.py`)
- [ ] A `sweep` CLI command (`cli.py`)

## License

No `LICENSE` file is included yet. Choose a license (for example MIT or Apache-2.0), add a `LICENSE` file, and
update the badge above before publishing.
````

## `docs/med-hammer-sim-flow.svg`

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="940" height="470" viewBox="0 0 940 470" font-family="system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,'DejaVu Sans',sans-serif" role="img" aria-labelledby="t d">
<title id="t">med-hammer-sim architecture and flow</title>
<desc id="d">Inputs (synthetic images, config and CLI flags, trigger spec) feed a simulation core (model with dormant Trojan, simulated bit flips, metrics runner), then checks (tensor checksums, activation z-score, scope guardrail), producing results.json, detection.json and console output. Bars show one sample run with seed 0.</desc>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8b949e"/></marker></defs>
<rect width="940" height="470" rx="16" fill="#0d1117"/>
<rect x="0.5" y="0.5" width="939" height="469" rx="16" fill="none" stroke="#30363d"/>
<text x="470" y="29" font-size="15" fill="#e6edf3" font-weight="bold" text-anchor="middle" >med-hammer-sim: simulated fault, dormant Trojan, baseline detection</text>
<text x="470" y="47" font-size="11" fill="#8b949e" font-weight="normal" text-anchor="middle" >Software-only simulation. Synthetic data. Bit flips on in-memory tensors.</text>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.2857;1" dur="0.70s" fill="freeze"/><text x="112.0" y="72" font-size="10" fill="#58a6ff" font-weight="bold" text-anchor="middle" letter-spacing="1.2">INPUTS</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.6429;1" dur="1.40s" fill="freeze"/><text x="350.0" y="72" font-size="10" fill="#a371f7" font-weight="bold" text-anchor="middle" letter-spacing="1.2">SIMULATION CORE</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.7619;1" dur="2.10s" fill="freeze"/><text x="600.0" y="72" font-size="10" fill="#3fb950" font-weight="bold" text-anchor="middle" letter-spacing="1.2">CHECKS AND GUARDRAILS</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8214;1" dur="2.80s" fill="freeze"/><text x="828.0" y="72" font-size="10" fill="#d29922" font-weight="bold" text-anchor="middle" letter-spacing="1.2">OUTPUTS</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.2857;1" dur="0.70s" fill="freeze"/><rect x="24" y="84" width="176" height="58" rx="10" fill="#161b22" stroke="#58a6ff" stroke-opacity="0.55"/><rect x="24" y="94" width="3" height="38" rx="1.5" fill="#58a6ff"/><text x="38" y="107" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Synthetic images</text><text x="38" y="124" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >data.py · synthetic</text><rect x="24" y="154" width="176" height="58" rx="10" fill="#161b22" stroke="#58a6ff" stroke-opacity="0.55"/><rect x="24" y="164" width="3" height="38" rx="1.5" fill="#58a6ff"/><text x="38" y="177" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Config + CLI flags</text><text x="38" y="194" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >config.py · cli.py</text><rect x="24" y="224" width="176" height="58" rx="10" fill="#161b22" stroke="#58a6ff" stroke-opacity="0.55"/><rect x="24" y="234" width="3" height="38" rx="1.5" fill="#58a6ff"/><text x="38" y="247" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Trigger spec</text><text x="38" y="264" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >trojan.py · trigger</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.6429;1" dur="1.40s" fill="freeze"/><rect x="240" y="80" width="220" height="212" rx="14" fill="#a371f7" fill-opacity="0.06" stroke="#a371f7" stroke-opacity="0.5" stroke-width="1.4"><animate attributeName="stroke-opacity" values="0.35;0.9;0.35" dur="3s" repeatCount="indefinite"/></rect><rect x="252" y="90" width="196" height="54" rx="10" fill="#161b22" stroke="#a371f7" stroke-opacity="0.55"/><rect x="252" y="100" width="3" height="34" rx="1.5" fill="#a371f7"/><text x="266" y="113" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Model + dormant Trojan</text><text x="266" y="130" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >model.py · trojan.py</text><rect x="252" y="158" width="196" height="54" rx="10" fill="#161b22" stroke="#a371f7" stroke-opacity="0.55"/><rect x="252" y="168" width="3" height="34" rx="1.5" fill="#a371f7"/><text x="266" y="181" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Simulated bit flips</text><text x="266" y="198" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >fault_injector.py</text><rect x="252" y="226" width="196" height="54" rx="10" fill="#161b22" stroke="#a371f7" stroke-opacity="0.55"/><rect x="252" y="236" width="3" height="34" rx="1.5" fill="#a371f7"/><text x="266" y="249" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Metrics runner</text><text x="266" y="266" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >eval.py</text>
<line x1="350" y1="144" x2="350" y2="157" stroke="#a371f7" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line><line x1="350" y1="212" x2="350" y2="225" stroke="#a371f7" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.7619;1" dur="2.10s" fill="freeze"/><rect x="500" y="80" width="200" height="212" rx="14" fill="#3fb950" fill-opacity="0.06" stroke="#3fb950" stroke-opacity="0.5" stroke-width="1.4"/><rect x="512" y="90" width="176" height="54" rx="10" fill="#161b22" stroke="#3fb950" stroke-opacity="0.55"/><rect x="512" y="100" width="3" height="34" rx="1.5" fill="#3fb950"/><text x="526" y="113" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Tensor checksums</text><text x="526" y="130" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >SHA-256 per tensor</text><rect x="512" y="158" width="176" height="54" rx="10" fill="#161b22" stroke="#3fb950" stroke-opacity="0.55"/><rect x="512" y="168" width="3" height="34" rx="1.5" fill="#3fb950"/><text x="526" y="181" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Activation z-score</text><text x="526" y="198" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >vs clean baseline</text><rect x="512" y="226" width="176" height="54" rx="10" fill="#161b22" stroke="#3fb950" stroke-opacity="0.55"/><rect x="512" y="236" width="3" height="34" rx="1.5" fill="#3fb950"/><text x="526" y="249" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Scope guardrail</text><text x="526" y="266" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >by design, tensors only</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8214;1" dur="2.80s" fill="freeze"/><rect x="740" y="84" width="176" height="58" rx="10" fill="#161b22" stroke="#d29922" stroke-opacity="0.55"/><rect x="740" y="94" width="3" height="38" rx="1.5" fill="#d29922"/><text x="754" y="107" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >results.json</text><text x="754" y="124" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >run: metrics, detection</text><rect x="740" y="154" width="176" height="58" rx="10" fill="#161b22" stroke="#d29922" stroke-opacity="0.55"/><rect x="740" y="164" width="3" height="38" rx="1.5" fill="#d29922"/><text x="754" y="177" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >detection.json</text><text x="754" y="194" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >detect: detectors only</text><rect x="740" y="224" width="176" height="58" rx="10" fill="#161b22" stroke="#d29922" stroke-opacity="0.55"/><rect x="740" y="234" width="3" height="38" rx="1.5" fill="#d29922"/><text x="754" y="247" font-size="12.5" fill="#e6edf3" font-weight="bold" text-anchor="start" >Console summary</text><text x="754" y="264" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >printed by the CLI</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.5455;1" dur="1.10s" fill="freeze"/><line x1="202" y1="113" x2="238" y2="113" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line><line x1="202" y1="183" x2="238" y2="183" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line><line x1="202" y1="253" x2="238" y2="253" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.7222;1" dur="1.80s" fill="freeze"/><line x1="462" y1="186" x2="498" y2="186" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8000;1" dur="2.50s" fill="freeze"/><line x1="702" y1="113" x2="738" y2="113" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line><line x1="702" y1="183" x2="738" y2="183" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line><line x1="702" y1="253" x2="738" y2="253" stroke="#8b949e" stroke-width="1.6" stroke-dasharray="6 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="12" to="0" dur="1s" repeatCount="indefinite"/></line></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><rect x="24" y="306" width="892" height="150" rx="12" fill="#0f141b" stroke="#21262d"/><text x="40" y="330" font-size="10" fill="#e6edf3" font-weight="bold" text-anchor="start" letter-spacing="1.2">SAMPLE RUN</text><text x="142" y="330" font-size="10.5" fill="#8b949e" font-weight="normal" text-anchor="start" >seed 0, default flags (fault-rate 0.001, trigger-size 3, target-label 1, 25 epochs)</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><text x="40" y="366" font-size="12" fill="#e6edf3" font-weight="normal" text-anchor="start" >ASR: trigger only</text><rect x="212" y="355" width="200" height="12" rx="6" fill="#21262d"/><text x="424" y="366" font-size="13" fill="#e6edf3" font-weight="bold" text-anchor="start" >0.00</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><text x="484" y="366" font-size="12" fill="#e6edf3" font-weight="normal" text-anchor="start" >ASR: trigger + fault</text><rect x="656" y="355" width="200" height="12" rx="6" fill="#21262d"/><rect x="656" y="355" width="200.0" height="12" rx="6" fill="#f85149"><animate attributeName="width" values="0;0;200.0" keyTimes="0;0.7083;1" dur="4.80s" fill="freeze"/></rect><text x="868" y="366" font-size="13" fill="#e6edf3" font-weight="bold" text-anchor="start" >1.00</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><text x="40" y="412" font-size="12" fill="#e6edf3" font-weight="normal" text-anchor="start" >Checksum recall</text><rect x="212" y="401" width="200" height="12" rx="6" fill="#21262d"/><rect x="212" y="401" width="200.0" height="12" rx="6" fill="#58a6ff"><animate attributeName="width" values="0;0;200.0" keyTimes="0;0.7200;1" dur="5.00s" fill="freeze"/></rect><text x="424" y="412" font-size="13" fill="#e6edf3" font-weight="bold" text-anchor="start" >1.00</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><text x="484" y="412" font-size="12" fill="#e6edf3" font-weight="normal" text-anchor="start" >Activation recall</text><rect x="656" y="401" width="200" height="12" rx="6" fill="#21262d"/><rect x="656" y="401" width="30.0" height="12" rx="6" fill="#d29922"><animate attributeName="width" values="0;0;30.0" keyTimes="0;0.7308;1" dur="5.20s" fill="freeze"/></rect><text x="868" y="412" font-size="13" fill="#e6edf3" font-weight="bold" text-anchor="start" >0.15</text></g>
<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8485;1" dur="3.30s" fill="freeze"/><text x="40" y="440" font-size="10" fill="#8b949e" font-weight="normal" text-anchor="start" >One run, not a benchmark. Values copied from results.json and detection.json (torch 2.15 CPU, Python 3.12).</text></g>
</svg>
```

## `docs/cli-demo.svg`

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="270" viewBox="0 0 900 270" font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','DejaVu Sans Mono','Courier New',monospace" role="img" aria-labelledby="t d">
<title id="t">med-hammer-sim CLI demo</title>
<desc id="d">Terminal showing python -m medhammer run and python -m medhammer detect with real output from seed 0.</desc>
<rect width="900" height="270" rx="12" fill="#0d1117"/>
<rect x="0.5" y="0.5" width="899" height="269" rx="12" fill="none" stroke="#30363d"/>
<path d="M0.5,34 L0.5,12.5 Q0.5,0.5 12.5,0.5 L887.5,0.5 Q899.5,0.5 899.5,12.5 L899.5,34 Z" fill="#161b22"/>
<line x1="0" y1="34.5" x2="900" y2="34.5" stroke="#30363d"/>
<circle cx="22" cy="17" r="6" fill="#ff5f56"/><circle cx="42" cy="17" r="6" fill="#ffbd2e"/><circle cx="62" cy="17" r="6" fill="#27c93f"/>
<text x="450.0" y="22" font-size="12" fill="#8b949e" font-weight="normal" text-anchor="middle" font-family="system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,'DejaVu Sans',sans-serif">med-hammer-sim</text>
<clipPath id="c1"><rect x="22" y="44" width="687" height="20"><animate attributeName="width" values="0;0;687" keyTimes="0;0.1818;1" dur="2.20s" fill="freeze"/></rect></clipPath><g clip-path="url(#c1)"><text x="22" y="58" font-size="13" xml:space="preserve"><tspan fill="#3fb950">$</tspan><tspan fill="#e6edf3"> python -m medhammer run --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1</tspan></text></g>
<text x="22" y="76" font-size="13" fill="#d29922" xml:space="preserve">SIMULATION ONLY: bit flips are applied to in-memory PyTorch tensors. No hardware<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="2.5s" fill="freeze"/></text>
<text x="22" y="94" font-size="13" fill="#d29922" xml:space="preserve">access, no exploit code, no real patient data, no network calls.<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="2.7s" fill="freeze"/></text>
<text x="22" y="112" font-size="13" fill="#e6edf3" xml:space="preserve">clean acc 1.000 -&gt; 1.000 (trojaned) | ASR trigger-only 0.000 | ASR trigger+fault 1.000<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="3.2s" fill="freeze"/></text>
<text x="22" y="130" font-size="13" fill="#8b949e" xml:space="preserve">wrote results.json<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="3.5s" fill="freeze"/></text>
<clipPath id="c2"><rect x="22" y="134" width="711" height="20"><animate attributeName="width" values="0;0;711" keyTimes="0;0.7049;1" dur="6.10s" fill="freeze"/></rect></clipPath><g clip-path="url(#c2)"><text x="22" y="148" font-size="13" xml:space="preserve"><tspan fill="#3fb950">$</tspan><tspan fill="#e6edf3"> python -m medhammer detect --seed 0 --fault-rate 0.001 --trigger-size 3 --target-label 1</tspan></text></g>
<text x="22" y="166" font-size="13" fill="#d29922" xml:space="preserve">SIMULATION ONLY: bit flips are applied to in-memory PyTorch tensors. No hardware<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="6.4s" fill="freeze"/></text>
<text x="22" y="184" font-size="13" fill="#d29922" xml:space="preserve">access, no exploit code, no real patient data, no network calls.<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="6.6s" fill="freeze"/></text>
<text x="22" y="202" font-size="13" fill="#e6edf3" xml:space="preserve">checksum P/R 1.00/1.00 | activation P/R 1.00/0.15<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="7.1s" fill="freeze"/></text>
<text x="22" y="220" font-size="13" fill="#8b949e" xml:space="preserve">wrote detection.json<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="7.4s" fill="freeze"/></text>
<text x="22" y="238" font-size="13" fill="#3fb950" opacity="0">$<animate attributeName="opacity" values="0;1" keyTimes="0;1" calcMode="discrete" dur="8s" fill="freeze"/></text>
<rect x="37.7" y="226" width="8" height="15" fill="#e6edf3" opacity="0"><animate attributeName="opacity" values="1;0" keyTimes="0;0.5" calcMode="discrete" dur="1s" begin="8s" repeatCount="indefinite"/></rect>
</svg>
```
