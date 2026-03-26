# AutoResearch — Project Memory

## Current Champion
- **Anchor**: P8-Lean 50K — 63.21% Top-1, 89.27% Top-3
- **Checkpoint**: `checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt`

## Known Bugs
- Auxiliary speed/role heads at 0% accuracy (debug before aux experiments)
- Switch prediction 37-48% vs 72-75% for moves (biggest accuracy lever)
- Calibration: systematic overconfidence in 0.4-0.8 range

## Some Example Experimental Variables
- Dataset size (50K used, >100K available)
- Switch-specific loss weighting
- Layers
- Hidden Dim
- Dropout
