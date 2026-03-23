# AutoResearch Implementation Plan

## Objective

This repository is organized around one task: **train and iterate on Phase 4 imitation models for Gen 3 OU**.

The implementation is intentionally centered on a small number of moving parts:

1. replay download,
2. replay processing,
3. feature construction,
4. BattleTransformer training,
5. experiment registration and evaluation.

## Core Components

### Data acquisition
- `scripts/download_replays.py`
- `scripts/download_replays_stratified.py`

### Processing pipeline
- `scripts/process_dataset.py`
- `src/data/replay_parser.py`
- `src/data/observation.py`
- `src/data/tensorizer.py`
- `src/data/dataset.py`
- `src/data/auxiliary_labels.py`
- `src/data/base_stats.py`
- `src/data/priors.py`
- `data/raw/pokedex/pokemon_base_stats.csv`

### Training stack
- `scripts/train_phase4.py`
- `scripts/train_p8_lean.py`
- `scripts/train_p4_25k.py`
- `src/models/battle_transformer.py`
- `src/environment/action_space.py`

### Experiment operations
- `Autoresearch/run_experiment.py`
- `Autoresearch/eval_harness.py`
- `Autoresearch/leaderboard.py`
- `Autoresearch/experiment_registry.json`

## Feature/Model Contract

### Observation rules
- Observations are first-person only.
- Hidden information is never exposed directly.
- Unknown information is represented as unknown, not leaked truth.
- Base stats are public knowledge once species is known and come from the included pokedex CSV.

### Transformer contract
- 14 tokens per step.
- 9-action policy space.
- legal mask always applied.
- optional auxiliary hidden-info prediction head.
- optional value head for larger configurations.

## Supported Training Jobs

### P8-Lean
Use `scripts/train_p8_lean.py` for the compact fast-iteration model:
- 3 layers
- 224 hidden dim
- 4 heads
- FFN multiplier 3x
- aux head on
- value head off

### P4
Use `scripts/train_p4_25k.py` for the larger Phase 4 model:
- 6 layers
- 384 hidden dim
- 6 heads
- aux head on
- value head on

## Operational Workflow

1. Download raw replay files into `data/raw/`.
2. Process them into `data/processed/`.
3. Validate the parser / observation / tensorizer / transformer tests.
4. Launch a baseline or experiment run.
5. Evaluate checkpoints and record results in `Autoresearch/`.

## Primary Validation Commands

```bash
pytest tests/test_parser.py tests/test_observation.py tests/test_tensorizer.py tests/test_transformer.py tests/test_base_stats.py
python scripts/train_p8_lean.py --dry-run --num-battles 10 --seeds 42
python scripts/train_p4_25k.py --dry-run --num-battles 10 --seeds 42
```

## Repository Design Principle

Everything in this repository should help an agent do one of the following:

- acquire data,
- process data,
- train a Phase 4 model,
- evaluate an experiment,
- understand how to run the pipeline correctly.
