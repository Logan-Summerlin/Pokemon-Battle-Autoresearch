# AutoResearch — Project Memory

## Current Champion
- **Anchor**: P8-Lean 50K — 63.21% Top-1, 89.27% Top-3
- **Checkpoint**: `checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt`

## Known Bugs
- Auxiliary speed/role heads at 0% accuracy (debug before aux experiments)
  - **Root cause documented**: see `important_fixes/001_auxiliary_head_missing_targets.md`
- Switch prediction 37-48% vs 72-75% for moves (biggest accuracy lever)
- Calibration: systematic overconfidence in 0.4-0.8 range

## Important Fixes
Check `important_fixes/` before starting experiments. Files marked HIGH priority should be resolved before running experiments that touch the affected component.

## Some Example Experimental Variables
- Dataset size (50K used, >100K available)
- Switch-specific loss weighting
- Layers
- Hidden Dim
- Dropout
