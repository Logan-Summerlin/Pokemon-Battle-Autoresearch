# AutoResearch — Project Memory

## Governing Plan
**Follow `Autoresearch/MASTER_RESEARCH_PLAN.md`.** Current phase: **A (Foundation repairs & rigor)**.
Phase A is blocking: complete A1 (checkpoint flags) and A2 (aux targets) and record the
noise floor (A3) before starting new experiments.

## Current Champion
- **AR-041** (`ar-041_t2_curr_w5_action_attn_s2`) — 67.79% Top-1, 93.42% Top-3, switch acc 60.51%
  - 5L/256d/4H, window=5, split_head + action_self_attention + move_identity + shuffle_moves,
    Elo-curriculum stage 2
- **Anchor (frozen reference)**: P8-Lean 50K — 63.21% Top-1, 89.27% Top-3
  - `checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt`

## Non-Negotiable Invariants (full text in MASTER_RESEARCH_PLAN.md §0)
1. **Move-identity conditioning + `shuffle_moves` always on.** The model chooses moves by
   identity (Solar Beam), never slot position (move2). Every eval includes the
   shuffled-moveset tripwire; collapse under shuffle = KILL. **All experiments predating
   the move-identity fix are contaminated — never use as baselines or evidence.**
2. **Hidden-information doctrine** — no omniscient features, explicit unknown markers;
   applies to self-play data too.
3. **Rated games only for imitation learning** — verify rating provenance before any
   dataset expansion (many undownloaded 1000–1500-bin Metamon battles are likely unrated).
4. **Registry-first** — hypothesis, parent, tier, one variable, KILL/RETRY/PROMOTE.

## Known Bugs (both HIGH — fix in Phase A before anything else)
- Checkpoints don't persist policy-head flags → eval_harness cannot load AR-041:
  `important_fixes/002_checkpoint_missing_head_flags.md`
- Auxiliary speed/role/move-family heads get zero gradient (0% accuracy):
  `important_fixes/001_auxiliary_head_missing_targets.md`
- Switch prediction 60.51% vs 71.86% for moves (biggest offline accuracy lever)
- Calibration: systematic overconfidence in 0.4–0.8 range

## Promotion Rules (summary)
- ≥2 seeds; delta must beat the recorded noise floor (Phase A3)
- Once online eval exists (Phase C): win-rate superiority over reigning champion,
  ≥400 paired battles, Wilson-CI significance; offline metrics become regression guards
- North star evolves: offline top-1 → gauntlet win rate → ladder GXE

## Important Fixes
Check `important_fixes/` before starting experiments. HIGH-priority files must be
resolved before running experiments that touch the affected component.
