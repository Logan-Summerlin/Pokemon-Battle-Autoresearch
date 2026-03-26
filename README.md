# Pokemon Battle AutoResearch

Autonomous research sandbox for **Phase 4 imitation learning** on Gen 3 OU Pokemon battles.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download and process data
python scripts/download_replays_stratified.py --target-size 100000 --output-dir data/raw
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --generation gen3ou

# Validate
pytest

# Train P8-Lean
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 --batch-size 64
```

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `src/data/` | Replay parsing, observations, tensorization, dataset loading |
| `src/models/` | BattleTransformer architecture |
| `src/environment/` | Action space definition |
| `scripts/` | Training entry points (`train_phase4.py`, `train_p8_lean.py`, `train_p4_25k.py`) |
| `Autoresearch/` | Experiment harness (run_experiment.py, eval_harness.py, leaderboard.py) |
| `data/` | Raw replays, processed tensors, vocabularies |
| `checkpoints/` | Model checkpoints |
| `tests/` | Unit tests |

## Autonomous Agent Setup

See `docs/RUNPOD_AUTORESEARCH_SETUP_GUIDE.md` for RunPod deployment. Agent instructions are in `CLAUDE.md`.
