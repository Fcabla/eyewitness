#!/usr/bin/env bash
# EYEWITNESS one-shot deploy to the hackathon org.
# Prereqs (Fernando, ~3 min): `hf auth login` (write token) and nothing else.
# Usage: bash deploy.sh [space-name]   (default: eyewitness)
set -euo pipefail
cd "$(dirname "$0")"

SPACE="build-small-hackathon/${1:-eyewitness}"

echo "==> Checking auth"
hf auth whoami || { echo "Run: hf auth login   (write token)"; exit 1; }

echo "==> Creating Space $SPACE (idempotent)"
hf repo create "$SPACE" --repo-type space --space_sdk gradio 2>/dev/null || true

echo "==> Uploading app"
hf upload "$SPACE" . . --repo-type space \
  --exclude ".venv/*" --exclude ".git/*" --exclude "__pycache__/*" \
  --exclude "tests/*" --exclude ".playwright-mcp/*" --exclude "*.log"

echo "==> IMPORTANT (manual, 1 min):"
echo "    1. Open https://huggingface.co/spaces/$SPACE/settings"
echo "    2. Hardware -> ZeroGPU"
echo "    3. Fill <TODO-FERNANDO-HF-USERNAME> in README.md (then re-run this script)"
echo "Done. Space: https://huggingface.co/spaces/$SPACE"
