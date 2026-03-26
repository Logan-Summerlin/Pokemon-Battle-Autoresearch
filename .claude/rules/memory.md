# AutoResearch — Project Memory

## What This Is
Automated research harness for perfecting Phase 4 (BattleTransformer imitation learning) of the Pokemon Battle Model. Targets >1300 Elo Gen 3 OU (ADV) singles from the Metamon dataset.

## Current Status
- **Anchor**: P8-Lean 50K — 63.21% Top-1, 89.27% Top-3
- **Phase**: AR-0 (Benchmark Lockdown)
- **Known bugs**: Aux speed/role heads at 0% accuracy; switch prediction 37-48% vs 72-75% for moves
- **Key variable**: Context window (current=2, optimal likely 2-8 turns)
- **Dataset**: >100K battles available, using as many as needed for >1300 Elo

## Agent Model
- **Single autonomous agent**: Claude Code owns the entire research loop
- No Codex agent. No human routing. No coordination overhead.
- Claude Code plans, implements, runs, evaluates, and records in a single tight loop.

## Autonomous Operation
- You are a fully autonomous research agent. NEVER stop to ask permission between experiments.
- Follow the experiment loop protocol religiously. After each experiment, GOTO step 1 immediately.
- The human will interrupt you when they want to — until then, keep going.
- During training runs, use 5-minute sleep/wake cycles (`sleep 300`) to prevent session timeout.
- Git state management: commit before experiments, revert on failure, advance on success.
- Crash recovery: fix once, then revert and move on. Never spend >10 minutes on a single crash.

## Non-Negotiable Rules
1. Hidden Information Doctrine — never train on features unavailable at decision time
2. Every experiment must be registered before running
3. Compare to parent experiment, not just anchor
4. Report move and switch accuracy separately
5. Run pytest after model/data code changes
6. Do not modify observation.py, tensorizer.py, or replay_parser.py without human approval

## Experiment Phases
- AR-0: Benchmark Lockdown (register anchor, eval harness)
- AR-1: Fix Known Issues (aux heads, legal mask, data dedup)
- AR-2: A40 Throughput Optimization (batch size, torch.compile, workers)
- AR-3: Data & Representation (window size, full dataset, loss weighting)
- AR-4: Architecture Tuning (scale up, dropout, hierarchical head)
- AR-5: Consolidation (combine best, 3-seed confirm, final report)

## File Ownership
Claude Code owns the entire approved edit surface:
- `Autoresearch/` — all files (configs, notes, results, harness scripts)
- `configs/` — YAML configuration files
- `scripts/train_phase4.py` — training loop
- `src/models/battle_transformer.py` — model architecture
- `src/data/dataset.py` — data loading and windowed dataset
- `src/data/auxiliary_labels.py` — auxiliary label construction
