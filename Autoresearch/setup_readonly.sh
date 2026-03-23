#!/bin/bash
# Setup read-only file permissions for the trimmed AutoResearch sandbox.
# Run once after cloning if you want a narrow agent edit surface.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Setting up read-only permissions in: $REPO_ROOT"

echo "  Setting src/ to read-only..."
find "$REPO_ROOT/src" -type f -exec chmod 444 {} \;

echo "  Setting scripts/ to read-only..."
find "$REPO_ROOT/scripts" -type f -exec chmod 444 {} \;

echo "  Setting tests/ to read-only..."
find "$REPO_ROOT/tests" -type f -exec chmod 444 {} \;

echo "  Setting docs/ to read-only..."
find "$REPO_ROOT/docs" -type f -exec chmod 444 {} \;

echo "  Setting data/ to read-only..."
find "$REPO_ROOT/data" -type f -exec chmod 444 {} \; 2>/dev/null || true

chmod 444 "$REPO_ROOT/pyproject.toml" 2>/dev/null || true
chmod 444 "$REPO_ROOT/README.md" 2>/dev/null || true
chmod 444 "$REPO_ROOT/CLAUDE.md" 2>/dev/null || true

echo "  Setting Autoresearch docs and harness helpers to read-only..."
for f in AGENTS.md README.md eval_harness.py leaderboard.py experiment_registry.json setup_readonly.sh; do
    chmod 444 "$REPO_ROOT/Autoresearch/$f" 2>/dev/null || true
done
find "$REPO_ROOT/Autoresearch/configs" -type f -exec chmod 444 {} \;
find "$REPO_ROOT/Autoresearch/notes" -type f -exec chmod 444 {} \;

echo "  Ensuring run_experiment.py is writable..."
chmod 644 "$REPO_ROOT/Autoresearch/run_experiment.py"

echo "  Ensuring output directories are writable..."
mkdir -p "$REPO_ROOT/Autoresearch/results" "$REPO_ROOT/checkpoints"
chmod 755 "$REPO_ROOT/Autoresearch/results" "$REPO_ROOT/checkpoints"

echo ""
echo "Done! File permissions set:"
echo "  WRITABLE: Autoresearch/run_experiment.py"
echo "  WRITABLE: Autoresearch/results/ (directory)"
echo "  WRITABLE: checkpoints/ (directory)"
echo "  READ-ONLY: Everything else"
