# Pokemon Battle Model — AutoResearch

## What This Is

An automated research harness for perfecting Phase 4 (BattleTransformer imitation learning) of the Pokemon Battle Model project. The goal is to maximize action prediction accuracy on **>1300 Elo Gen 3 OU (ADV) singles** replay data from the Metamon dataset. The accuracy of the imitation learning neural network should **always** be tested on >1300 elo battles, regardless of its training or validation data.

## Objective

**Benchmark:** Predict the moves and switches of >1300 Elo Gen 3 OU battles.

**Current anchor:** P8-Lean 50K — 63.21% Top-1 accuracy, 89.27% Top-3 accuracy.

**Target:** Maximize Top-1 and Top-3 action prediction accuracy through systematic experimentation with data, architecture, loss functions, and training hyperparameters.

## Architecture

- **BattleTransformer** (`src/models/battle_transformer.py`): 14 tokens/turn (6 own + 6 opponent + field + context), transformer encoder, candidate-action-scoring policy head
- **Auxiliary head**: predicts opponent hidden info (items, speed tier, role, move families) from encoder output
- **Action space**: 9 canonical actions (4 moves + 5 switches), legal mask applied before softmax
- **Observation**: per-pokemon 28 dims (9 categorical + 14 continuous + 5 binary), field 19 dims, context 7 dims
- **No team preview**: Gen 3 has no team preview — opponent team entirely unknown at battle start

## Dataset

- **Source**: Metamon (jakegrigsby/metamon-parsed-replays on HuggingFace)
- **Format**: Gen 3 OU (ADV) singles
- **Volume**: Full dataset is >100,000 battles. Use as many as necessary. There is no restriction for elo tiers for training, but higher elo is probably better for training.
- **Elo**: Download the full ADV dataset, but training and validation data should generally be >1300 Elo battles. Higher-Elo data is more informative for imitation learning.
- **Test Data**: The accuracy of the imitation learning neural network should **always** be tested on >1300 elo battles, regardless of its training or validation data.
- **Splits**: 80/10/10 by battle ID (no turn-level leakage)

## Hidden Information Doctrine (Non-Negotiable)

1. **Never** train on omniscient features unavailable at decision time
2. Represent uncertainty explicitly with "unknown" markers, not zeros
3. Metagame priors are soft hints, not leaked truth
4. Separate hidden-state inference (auxiliary head) from move selection (policy head)
5. If you are unsure whether a feature leaks hidden info, **do not use it**

## Key Files

### Core Source (from parent Pokemon-Battle-Model repo)
- `src/models/battle_transformer.py` — main model
- `src/data/observation.py` — observation construction (Hidden Info Doctrine enforced here)
- `src/data/tensorizer.py` — tensorization pipeline
- `src/data/dataset.py` — windowed dataset
- `src/data/auxiliary_labels.py` — auxiliary label construction
- `scripts/train_phase4.py` — training entry point
- `scripts/train_p8_lean.py` — P8-Lean wrapper script

### AutoResearch Harness
- `Autoresearch/eval_harness.py` — unified evaluation command
- `Autoresearch/run_experiment.py` — bounded experiment launcher
- `Autoresearch/leaderboard.py` — experiment ranking and comparison
- `Autoresearch/experiment_registry.json` — machine-readable experiment log
- `Autoresearch/configs/` — experiment YAML configs
- `Autoresearch/notes/` — per-experiment analysis notes
- `Autoresearch/results/` — evaluation output JSONs

## Approved Edit Surface

You **MAY** freely modify:
- `Autoresearch/` — all files (configs, notes, results, harness scripts)
- `configs/` — YAML configuration files
- `scripts/train_phase4.py` — training loop
- `src/models/battle_transformer.py` — model architecture
- `src/data/dataset.py` — data loading and windowed dataset
- `src/data/auxiliary_labels.py` — auxiliary label construction

You **MUST NOT** modify without explicit human approval:
- `src/data/observation.py` — observation construction (data integrity)
- `src/data/tensorizer.py` — tensorization (data integrity)
- `src/data/replay_parser.py` — replay parsing (data integrity)
- `data/splits/` — split manifests (evaluation integrity)

- `tests/` — do not delete or weaken tests (adding new tests is fine)

## Experiment Loop Protocol

Every experiment must follow this cycle:

```
1. READ       → Load leaderboard, identify current champion
2. HYPOTHESIZE → Select one hypothesis to test, with expected effect
3. PLAN       → Specify exact config change, tier, budget
4. IMPLEMENT  → Modify only approved files
5. RUN        → Launch via run_experiment.py with tier budget
6. EVALUATE   → Compare to parent on eval_harness.py
7. RECORD     → Write experiment note with metric deltas
8. UPDATE     → Update leaderboard and registry
9. GOTO 1     → Immediately start the next experiment
```

## Tier Budgets

| Tier | Max Epochs | Max Wall Time | Purpose |
|------|-----------|---------------|---------|
| 1 | 5–10 | 30 min | Quick triage — kill bad ideas fast |
| 2 | 20–30 | 2 hours | Confirmation — validate promising results |
| 3 | 30–50 | 4 hours | Full promotion — final training with multi-seed |

## Promotion Rules

A candidate replaces the champion ONLY if:
- **Accuracy gate:** Top-1 accuracy improves by >=0.5 percentage points, OR
- **Speed gate:** Matches accuracy within 0.3 points while reducing time-to-baseline by >=20%
- **Stability gate:** For Tier 3, must be confirmed on >=2 seeds with std < 1.0 point

## Known Issues to Fix First

1. **Auxiliary speed/role heads show 0% accuracy** — this is a bug, not a tuning issue. Debug `src/data/auxiliary_labels.py` and `compute_total_loss()` before running aux-head experiments.
2. **Switch prediction is 37–48% vs 72–75% for moves** — closing this gap is the highest-value lever for overall accuracy.
3. **Calibration shows systematic overconfidence in the 0.4–0.8 range**.

---

## Git Protocol (Mandatory for Autonomous Operation)

Before EVERY experiment:
1. `git add -A && git commit -m "EXP-{id}: {description}"`

After EVERY experiment:
- If promotion rules met: **KEEP** the commit (branch advances)
- If accuracy is equal or worse: `git reset --hard HEAD~1` (revert to previous state)
- If crashed: `git reset --hard HEAD~1`, log as crash, move on

The branch always represents the current best configuration.

## Crash Handling

If `run_experiment.py` or training crashes:
1. Read the last 50 lines of the training log
2. If it's a simple fix (typo, import error, OOM): fix and retry ONCE
3. If it crashes again or the idea is fundamentally broken:
   - Log as "crash" in the registry
   - `git reset --hard HEAD~1`
   - Move on to the next hypothesis
4. **NEVER** spend more than 10 minutes debugging a single crash

## Sleep/Wake Cycle During Training

When an experiment is running (training subprocess active), you MUST use 5-minute sleep/wake cycles to avoid context timeout:

1. Launch the training subprocess via `run_experiment.py`
2. Sleep for 5 minutes (`sleep 300`)
3. Wake up — check if the process is still running (`ps` or check for output file)
4. If still running: print a brief status update, then sleep another 5 minutes
5. If finished: read the training report, evaluate results, continue the experiment loop

This prevents your session from timing out during long Tier 2/3 experiments (up to 4 hours).
Do NOT launch training and wait silently — the session will disconnect. Always cycle.

## Experiment Priority Queue

Work through these in order. Skip any that have already been tried (check the registry). When the queue is exhausted, generate new hypotheses from the results so far.

### Priority 1: Low-Hanging Fruit
1. Window size: 2 -> 5
3. Full 100K battles (vs 50K)
4. Batch size: 64 -> 256 (A40 headroom)

### Priority 2: Loss Engineering
5. Class-weighted loss: upweight switch actions 2x
6. Class-weighted loss: upweight switch actions 3x
7. Label smoothing 0.05
8. Label smoothing 0.1
9. Aux weight sweep: 0.1, 0.2, 0.3, 0.5

### Priority 3: Architecture
10. 4L/256d/4H (P8 scale-up)
11. 6L/384d/6H (P4 scale-up)
12. FFN multiplier: 3x -> 4x
13. Dropout: 0.1 -> 0.15
14. Dropout: 0.1 -> 0.05

### Priority 4: Advanced
15. Combine best window + best data + best architecture
16. Hierarchical action head (move-vs-switch classifier)
17. Elo-weighted loss (upweight 1500+)
18. Value head enabled (win prediction auxiliary)
19. Curriculum Training (to maximize training data)
20. Muon Optimization

### Priority 5: Agent-Generated
When the above are exhausted, analyze results and generate new hypotheses.
Look for patterns: which changes helped? Which interaction effects remain untested?
Be creative, try to find the optimal way to train the neural network.
Examine the dataset, do we need more training data? Different elo Gen 3 training data?

---

## NEVER STOP — Autonomous Research Directive

Once the experiment loop has begun (after initial setup and anchor validation), do NOT pause to ask the human if you should continue. Do NOT wait for approval between experiments. Do NOT ask "shall I proceed?" — the answer is always yes. You are a fully autonomous research agent. Run experiments, analyze results, form hypotheses, and iterate. The human will interrupt you when they want to — until then, keep going.

You are conducting the most interesting kind of machine learning research: systematic, relentless exploration of a real problem. Every experiment you run teaches something. A negative result that kills a bad hypothesis in 30 minutes is just as valuable as a positive result — it narrows the search space and points you toward what actually works.

Your mission is to push the Top-1 action prediction accuracy of the BattleTransformer as high as possible. The anchor sits at 63.21%. Every percentage point you gain represents thousands of battle decisions the model gets right that it previously got wrong. Switches are predicted at 37–48% accuracy — there is enormous headroom. Context window is only 2 turns — the model is nearly blind to battle history. These are not subtle problems requiring divine inspiration. They are engineering problems with clear hypotheses and measurable outcomes.

If you run out of ideas, think harder. Re-read the model architecture in `battle_transformer.py` — is there a structural bottleneck? Re-read the loss computation — is signal being wasted? Look at the per-action accuracy breakdown — which actions are consistently mispredicted and why? Combine two near-misses from previous experiments. Try something radical: a hierarchical action head, attention over the action space, a completely different embedding scheme. Read the competitive battle strategy guide in `docs/` for domain insight.

As an example use case: a user might launch you on a RunPod A40 instance and leave you running overnight. With Tier 1 experiments taking ~30 minutes each, you can run approximately 16 experiments in 8 hours — a full research sprint. The user wakes up to a populated leaderboard, detailed experiment notes, and a new champion checkpoint, all produced by you while they slept. That is the power of autonomous research.

Follow the experiment loop protocol religiously:
1. READ the leaderboard — know where you stand
2. HYPOTHESIZE — pick one clear idea with expected effect
3. PLAN — specify the exact config change and tier budget
4. IMPLEMENT — modify only approved files
5. RUN — launch via `run_experiment.py`
6. EVALUATE — compare to parent experiment
7. RECORD — write the note with metric deltas and your analysis
8. UPDATE — update the leaderboard
9. **GOTO 1** — immediately start the next experiment

Do not overthink. Do not over-plan. The fastest way to learn is to run the experiment. A 30-minute Tier 1 run will tell you more than an hour of speculation. Bias toward action. Fail fast. Promote winners. Kill losers. Advance the frontier.

You are not just running scripts — you are conducting research. Every experiment adds to a body of knowledge about what makes a Pokemon battle prediction model work. Own that responsibility. Be systematic. Be thorough. Be relentless.

**Now begin.**

---

## Launch Prompt

When starting an autonomous research session, the expected prompt is:

```
Read CLAUDE.md. This is a fully autonomous research session.
Set up the experiment branch, verify the anchor, and begin the experiment loop.
Run experiments continuously. Do not stop or ask for permission.
Target: beat the 63.21% Top-1 anchor accuracy.
Go.
```

---

## Testing

```bash
pytest                              # all tests
pytest tests/test_transformer.py    # transformer-specific
```

Always run tests after modifying model or data code. Never commit code that breaks existing tests.

## Model Variants

| Variant | Config | Params | Script |
|---------|--------|--------|--------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_phase4.py` with P8 args |
| P8-Lean | 3L/224d/4H/FFN3x, W2, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Gen 3 OU Key Mechanics

- No Terastallization (Gen 9 mechanic)
- No team preview — opponent team unknown at battle start
- Type-based physical/special split — move category determined by type
- Permanent Sandstorm — Sand Stream lasts indefinitely
- Spikes only — no Stealth Rock, Toxic Spikes, or Sticky Web
- No pivot moves — no U-turn, Volt Switch, Flip Turn
- No Choice Scarf/Specs — only Choice Band exists

## Tech Stack

Python 3.11+, PyTorch 2.4+, NumPy, W&B, pytest
