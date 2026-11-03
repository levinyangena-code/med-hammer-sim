# Copilot instructions for med-hammer-sim

This repository is a SIMULATION-ONLY, defensive red-team proof of concept.

## Hard rules
- Never add real Rowhammer or other hardware-exploit code, hardware or physical-memory access,
  malware, network calls, or real/downloaded patient data.
- Bit flips are applied only to in-memory PyTorch tensors (copies or model parameters).

## Conventions
- Python 3.11+, type hints on every function signature.
- Google-style docstrings with Args / Returns / Raises.
- Deterministic: every random operation takes a seed or a seeded generator.
- Config and result schemas use pydantic v2; the CLI uses typer.
- Tests use pytest, must be deterministic, and must pass with plain `pytest`.
- Keep edits small, run `pytest` after changes, and update the README if behaviour changes.
