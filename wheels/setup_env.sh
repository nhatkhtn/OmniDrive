#!/usr/bin/env bash
# Single entry point for setting up the OmniDrive env. Always use this
# instead of a bare `uv sync` -- see "Why this script" in README.md: the
# vendored mmcv-full wheel, peft, sentencepiece and the cu11 runtime shim
# aren't declared in pyproject.toml/uv.lock, so `uv sync` (its normal
# exact-sync behavior) silently prunes them back out on every re-sync, not
# just the first one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIDRIVE_ROOT="$(dirname "$HERE")"

: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT to a node-local path first, e.g. export UV_PROJECT_ENVIRONMENT=/tmp/.venv-omnidrive}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/.uv_cache}"

echo "[setup_env] uv sync --frozen (this installs the *wrong* mmcv-full -- expected, overridden next)"
uv sync --project "$OMNIDRIVE_ROOT" --frozen

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

echo "[setup_env] overriding mmcv-full with the vendored torch-2.7/cu128 build"
uv pip install --force-reinstall --no-deps \
  "$HERE/mmcv_full-1.7.0-cp39-cp39-linux_x86_64.whl" "numpy==1.23.4"

echo "[setup_env] installing peft/sentencepiece + cu11 runtime shim (missing from the lock)"
uv pip install "peft>=0.12,<0.13" "sentencepiece>=0.2.1" \
  "nvidia-cuda-runtime-cu11==11.7.99"

echo "[setup_env] done. Run with:"
echo "  LD_LIBRARY_PATH=\"$UV_PROJECT_ENVIRONMENT/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH\" \\"
echo "    uv run --no-sync --project $OMNIDRIVE_ROOT python run_sample_omnidrive.py"
