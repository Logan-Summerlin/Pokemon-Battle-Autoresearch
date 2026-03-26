# AutoResearch — Single-Agent Autonomous Operation Guide

This document defines the operating model for fully autonomous Claude Code research on the Pokemon Battle Model.

---

## Agent Model

**Claude Code is the sole autonomous agent.** There is no second agent, no human routing, no coordination overhead. Claude Code owns the entire research loop:

| Responsibility | Owner |
|---------------|-------|
| Experiment planning & hypothesis generation | Claude Code |
| Config generation | Claude Code |
| Code changes (model, data, loss) | Claude Code |
| Script maintenance (run_experiment.py, etc.) | Claude Code |
| Training execution & monitoring | Claude Code |
| Log parsing & metric extraction | Claude Code |
| Result analysis & experiment notes | Claude Code |
| Leaderboard updates | Claude Code |
| Bug investigation | Claude Code |
| Git state management (commit/revert) | Claude Code |

---

## Experiment Loop

Every experiment follows this cycle. **Never break the loop. Never pause for approval.**

```
1. READ       → Load leaderboard (python Autoresearch/leaderboard.py)
2. HYPOTHESIZE → Pick one hypothesis from priority queue or generate new
3. PLAN       → Specify exact config change, expected effect, tier budget
4. IMPLEMENT  → Modify only approved files; git commit before training
5. RUN        → Launch via run_experiment.py (use sleep/wake cycles)
6. EVALUATE   → Compare to parent on eval_harness.py
7. RECORD     → Write experiment note with metric deltas + analysis
8. UPDATE     → Update leaderboard and registry
9. DECIDE     → PROMOTE (keep commit) or KILL (git reset --hard HEAD~1)
10. GOTO 1    → Immediately start next experiment
```

---

## Git State Management

The git branch always represents the current best configuration.

**Before every experiment:**
```bash
git add -A && git commit -m "EXP-{id}: {description}"
```

**After every experiment:**
- Promotion rules met → **KEEP** commit (branch advances)
- No improvement → `git reset --hard HEAD~1` (revert)
- Crash → `git reset --hard HEAD~1` + log as crash

---

## Crash Recovery Protocol

1. Read the last 50 lines of the training log
2. Simple fix (typo, import, OOM) → fix and retry **ONCE**
3. Second crash or fundamentally broken → log as "crash", revert, move on
4. **NEVER** spend more than 10 minutes debugging a single crash

---

## Sleep/Wake Cycles

During training runs, use 5-minute cycles to avoid session timeout:

1. Launch training subprocess
2. `sleep 300`
3. Check if process is running (`ps` or check output file)
4. Still running → brief status update → `sleep 300` again
5. Finished → read report → continue loop

---

## Experiment Note Template

Every experiment gets a note in `Autoresearch/notes/{id}_{name}.md`:

```markdown
# {experiment_id}: {name}

## Hypothesis
{What you expected and why}

## Change
{Exact config/code change made}

## Results
| Metric | Parent | This | Delta |
|--------|--------|------|-------|
| Top-1  |        |      |       |
| Top-3  |        |      |       |
| NLL    |        |      |       |
| Move   |        |      |       |
| Switch |        |      |       |
| ECE    |        |      |       |
| Time   |        |      |       |

## Analysis
{Why did this work or not work?}

## Decision
{PROMOTE / KILL / RETRY at higher tier}
```

---

## Restrictions

1. **Hidden Information Doctrine** — never train on features unavailable at decision time
2. **Do not modify** `observation.py`, `tensorizer.py`, `replay_parser.py` without human approval
3. **Do not delete or weaken tests** — adding new tests is fine
4. **Register before running** — every experiment gets a registry entry before training starts
5. **Compare to parent** — not just the anchor, but the direct parent experiment
6. **Report move and switch accuracy separately** — always
7. **Run pytest** after modifying model or data code

---

## Approved Edit Surface

**MAY freely modify:**
- `Autoresearch/` — all files
- `configs/` — YAML configs
- `scripts/train_phase4.py` — training loop
- `src/models/battle_transformer.py` — model architecture
- `src/data/dataset.py` — data loading
- `src/data/auxiliary_labels.py` — aux label construction

**MUST NOT modify:**
- `src/data/observation.py` — observation construction
- `src/data/tensorizer.py` — tensorization pipeline
- `src/data/replay_parser.py` — replay parsing
- `data/splits/` — split manifests
- `docs/EVALUATION_SPEC.md` — benchmark definition
- `tests/` — do not delete or weaken (adding is fine)
