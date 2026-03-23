# Pokemon Battle AutoResearch

A standalone repository for **Phase 4 imitation learning** on Gen 3 OU Pokemon battles.

This codebase contains the full training pipeline needed to:

- download replay data from the Metamon dataset,
- process raw replays into first-person supervised-learning tensors,
- train BattleTransformer imitation models,
- run the supported `P8-Lean` and `P4` training jobs, and
- manage bounded experiment loops through the `Autoresearch/` harness.

## Repository Scope

### Data acquisition
- `scripts/download_replays.py` — general replay download + sampling.
- `scripts/download_replays_stratified.py` — stratified Gen 3 OU download plan.
- `scripts/process_dataset.py` — raw replay → processed tensor pipeline.

### Training entry points
- `scripts/train_phase4.py` — main Phase 4 training loop.
- `scripts/train_p8_lean.py` — wrapper for the lean Phase 4 model.
- `scripts/train_p4_25k.py` — wrapper for the larger Phase 4 model.

### Model and feature pipeline
- `src/data/replay_parser.py` — parses Metamon replay files.
- `src/data/observation.py` — builds hidden-information-safe first-person observations.
- `src/data/tensorizer.py` — tensorizes battle state into model features.
- `src/data/dataset.py` — processed dataset save/load helpers.
- `src/data/auxiliary_labels.py` — auxiliary hidden-info labels.
- `src/data/base_stats.py` — species base-stat lookup.
- `src/data/priors.py` — metagame prior aggregation used during processing.
- `src/models/battle_transformer.py` — BattleTransformer architecture and losses.
- `src/environment/action_space.py` — canonical 9-action vocabulary.
- `data/raw/pokedex/pokemon_base_stats.csv` — base-stat source used by the feature pipeline.

### Experiment harness
- `Autoresearch/run_experiment.py`
- `Autoresearch/eval_harness.py`
- `Autoresearch/leaderboard.py`
- `Autoresearch/experiment_registry.json`
- `Autoresearch/configs/anchor.yaml`
- `Autoresearch/notes/000_anchor.md`

### Documentation
- `docs/AUTORESEARCH_IMPLEMENTATION_PLAN.md`
- `docs/RUNPOD_AUTORESEARCH_SETUP_GUIDE.md`
- `data/processed/DATA_README.md`
- `Autoresearch/README.md`
- `CLAUDE.md` and `AGENTS.md` instructions remain in place.

## BattleTransformer Summary

The retained Phase 4 model is a structured transformer for action prediction under partial observability.

- **Format:** Gen 3 OU (ADV) singles.
- **Action space:** 9 canonical actions (4 moves + 5 switches).
- **Tokens per step:** 14 total.
  - 6 own-team slots
  - 6 opponent-team slots
  - 1 field token
  - 1 context token
- **Feature layout:**
  - pokemon features: 9 categorical + 14 continuous + 5 binary
  - field features: 19 dims
  - context features: 7 dims
- **Heads:**
  - policy head for action prediction
  - auxiliary head for hidden-info targets
  - optional value head in the larger configuration

## End-to-End Pipeline

1. **Download replays** with `scripts/download_replays.py` or `scripts/download_replays_stratified.py`.
2. **Process raw files** with `scripts/process_dataset.py`.
3. **Load processed tensors** through the Phase 4 training stack.
4. **Train** with `scripts/train_phase4.py`, `scripts/train_p8_lean.py`, or `scripts/train_p4_25k.py`.
5. **Evaluate and compare** experiments through `Autoresearch/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Download data
python scripts/download_replays_stratified.py --target-size 100000 --output-dir data/raw

# 2. Process data
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --generation gen3ou

# 3. Validate the retained Phase 4 stack
pytest tests/test_parser.py tests/test_observation.py tests/test_tensorizer.py tests/test_transformer.py tests/test_base_stats.py

# 4. Train P8-Lean
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 --batch-size 64

# 5. Train P4
python scripts/train_p4_25k.py --num-battles 25000 --seeds 42 --batch-size 32
```

## Data Notes

- `data/processed/vocabs/` contains the vocabularies used by the feature pipeline.
- `data/processed/vocabs/gen3ou/` is also preserved.
- Large replay tensors are not committed; see `data/processed/DATA_README.md` for the expected processed layout.

## Next Docs to Read

- `docs/AUTORESEARCH_IMPLEMENTATION_PLAN.md`
- `docs/RUNPOD_AUTORESEARCH_SETUP_GUIDE.md`
- `Autoresearch/README.md`
