# RunPod AutoResearch Setup Guide

## Recommended Instance

- **GPU:** A40 48GB or similar.
- **CPU:** 8+ vCPUs.
- **RAM:** 32GB+.
- **Disk:** 100GB persistent volume.
- **Base image:** Python 3.11 with CUDA-enabled PyTorch.

## 1. Clone and install

```bash
cd /workspace
git clone <your-repo-url> Pokemon-Battle-Autoresearch
cd Pokemon-Battle-Autoresearch

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Download raw replay data

### General sampler
```bash
python scripts/download_replays.py --generation gen3ou --sample-size 10000 --elo-threshold 1300 --output-dir data/raw
```

### Stratified Gen 3 OU sampler
```bash
python scripts/download_replays_stratified.py --target-size 100000 --output-dir data/raw
```

## 3. Process replays

```bash
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --generation gen3ou
```

## 4. Validate the stack

```bash
pytest tests/test_parser.py tests/test_observation.py tests/test_tensorizer.py tests/test_transformer.py tests/test_base_stats.py
python scripts/train_p8_lean.py --dry-run --num-battles 10 --seeds 42
python scripts/train_p4_25k.py --dry-run --num-battles 10 --seeds 42
```

## 5. Launch training

### P8-Lean
```bash
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 --batch-size 64
```

### P4
```bash
python scripts/train_p4_25k.py --num-battles 25000 --seeds 42 --batch-size 32
```

## 6. Run experiment management

```bash
python Autoresearch/run_experiment.py \
  --name window5 \
  --parent anchor \
  --tier 1 \
  --config-override max_window=5

python Autoresearch/leaderboard.py
```
