#!/bin/bash
# Reproduce headline results from the thesis.
# Must regenerate ≥80% of headline numbers within 10% on the same hardware.
# Usage: bash scripts/reproduce_results.sh

set -euo pipefail

echo "=== Reason-Reduce Results Reproduction ==="
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo 'not-a-repo')"
echo "Date: $(date -u)"
echo ""

# Run smoke first
echo "[1/4] Smoke test..."
python scripts/smoke.py
echo "  ✓ Smoke test passed"

# Run unit tests
echo "[2/4] Unit tests..."
python -m pytest tests/unit -x --timeout=30 -q
echo "  ✓ Unit tests passed"

# Run integration tests
echo "[3/4] Integration tests..."
python -m pytest tests/integration -x --timeout=60 -q
echo "  ✓ Integration tests passed"

# Run benchmarks (TODO[Phase-5])
echo "[4/4] Benchmarks..."
echo "  → TODO: Full benchmark suite (Phase 5)"

echo ""
echo "=== Reproduction Complete ==="
