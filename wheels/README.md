# Vendored wheels

## Recreating the working OmniDrive env from scratch

1. Get an allocation on a node with a Blackwell/B200 GPU (sm_100 CUDA
   compute capability) from your scheduler. If it's Slurm, always pass
   `--mem` explicitly on job steps -- a default per-step memory cgroup can
   be too small and will silently OOM-kill the 7B model load.
2. Venv must live in `$TMPDIR` (fast local scratch), never on a shared/
   quota-limited project filesystem -- and if `$TMPDIR` is node-local,
   this setup has to be redone per node/allocation:
   ```bash
   export UV_PROJECT_ENVIRONMENT=/tmp/.venv-omnidrive
   export UV_CACHE_DIR=/tmp/.uv_cache
   ```
3. Run `wheels/setup_env.sh` -- **not** a bare `uv sync`. It runs
   `uv sync --project OmniDrive --frozen` (which installs the base env:
   torch 2.7.0+cu128, flash-attn 2.8.0, transformers, and builds
   `mmdet3d`/`openlanev2` from source -- also installs the *wrong*
   `mmcv-full`, see below, expected) and then force-installs the vendored
   wheel + the packages missing from the lock on top (see "Why this script"
   below for why this can't just live in the lockfile):
   ```bash
   ./wheels/setup_env.sh
   ```
4. Run, with the cu11 runtime shim on `LD_LIBRARY_PATH`:
   ```bash
   LD_EXTRA=/tmp/.venv-omnidrive/lib/python3.9/site-packages/nvidia/cuda_runtime/lib
   LD_LIBRARY_PATH="$LD_EXTRA:$LD_LIBRARY_PATH" \
     uv run --no-sync --project OmniDrive python run_sample_omnidrive.py
   ```

The trained checkpoint (`iter_10548.pth`) is loaded from the path
`CHECKPOINT_PATH` hardcodes in `run_sample_omnidrive.py` -- update that
constant if your checkpoint lives somewhere else.

## mmcv_full-1.7.0-cp39-cp39-linux_x86_64.whl

No prebuilt `mmcv-full==1.7.0` wheel exists for torch 2.7+cu128 (the OpenMMLab
wheel index only covers up to cu117/torch1.13), and the fork's env needs
torch 2.7+cu128 for Blackwell/B200 GPUs.

Built with:
- torch 2.7.0+cu128, CUDA 12.8.1 (`nvcc` from the CUDA 12.8 toolkit install)
- `TORCH_CUDA_ARCH_LIST="10.0+PTX"` (sm_100 = B200; `+PTX` keeps forward-compat
  via JIT for future archs)
- Python 3.9 (`cp39`)
- `MMCV_WITH_OPS=1 FORCE_CUDA=1`

mmcv-full 1.7.0's `setup.py` hardcodes `-std=c++14` for its CUDA/C++
extensions; torch 2.7's headers require C++17. The sdist needs a one-line
patch before it will compile:

```bash
sed -i "s/c++14/c++17/g" setup.py
```

## Why this script (manual override -- NOT wired into uv.lock)

`pyproject.toml`/`uv.lock` still point `mmcv-full` at the openmmlab cu117
registry (the only thing that resolves cleanly with `uv sync --frozen`).
Pointing `[tool.uv.sources]` at this local wheel instead was tried and
reverted: it requires an explicit `wheels = [{ path, hash }]` entry in
`uv.lock` (a plain `path` source isn't enough -- uv rejects it as
"incompatible with the current platform" without one), and adding that
entry, even alongside a correctly-installed venv, made a *subsequent*
`uv sync --frozen` fail elsewhere (a pre-existing, unrelated issue: uv's
build isolation for the `mmdet3d` editable install sometimes can't see
`torch` on a second sync, independent of anything mmcv-related). Not worth
fixing to auto-wire a one-line override.

`peft` and `sentencepiece` (used by `projects/mmdet3d_plugin`) are missing
from `pyproject.toml` for the same underlying reason -- adding them requires
a full `uv lock`, which hits the same `mmdet3d` build-isolation issue.

**Footgun this creates:** none of the above (mmcv, peft, sentencepiece, the
cu11 shim) are declared in the lockfile, so `uv sync`'s normal exact-sync
behavior silently *prunes them back out* on every re-sync -- not just the
first one. Re-running a bare `uv sync --frozen` on an already-working venv
(e.g. after touching `pyproject.toml` for an unrelated reason) quietly
breaks it again with no warning at sync time.

`wheels/setup_env.sh` exists specifically so nobody has to remember this: it
always runs `uv sync --project OmniDrive --frozen` immediately followed by
the override installs, as one atomic step. **Always run that script, never
a bare `uv sync`, for this project.** As a second line of defense,
`run_sample_omnidrive.py` also calls `check_env()` at the top of `main()`,
which fails fast with a pointer back to this script if `mmcv.ops`, `peft`,
or `sentencepiece` don't import cleanly (e.g. because someone bypassed the
script and the prune silently happened).

For reference, what the script does under the hood, `--no-deps` on the mmcv
install so it doesn't drag `numpy` off its pin:

```bash
export VIRTUAL_ENV=/tmp/.venv-omnidrive   # or wherever the env lives
uv pip install --force-reinstall --no-deps \
  OmniDrive/wheels/mmcv_full-1.7.0-cp39-cp39-linux_x86_64.whl "numpy==1.23.4"
uv pip install "peft>=0.12,<0.13" "sentencepiece>=0.2.1" \
  "nvidia-cuda-runtime-cu11==11.7.99"
```

At runtime, the compiled `.so` needs `libcudart.so.11.0` on `LD_LIBRARY_PATH`
(see below) -- cu128's torch wheels don't ship it.

## Rebuilding (e.g. for a different Python/CUDA/arch combo)

Needs a GPU node (arch-specific `nvcc` codegen) with torch 2.7+cu128 already
installed in the target venv.

```bash
export CUDA_HOME=/path/to/cuda-12.8   # match your cluster's CUDA install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="10.0+PTX"          # B200 = sm_100
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=4
export VIRTUAL_ENV=/path/to/venv-with-torch-2.7-cu128

pip download --no-binary=:all: --no-deps --no-build-isolation \
  "mmcv-full==1.7.0" -d /tmp/mmcv_build
cd /tmp/mmcv_build && tar -xzf mmcv-full-1.7.0.tar.gz
sed -i "s/c++14/c++17/g" mmcv-full-1.7.0/setup.py
pip wheel --no-build-isolation --no-deps -w /tmp/mmcv_wheel_out \
  mmcv-full-1.7.0/
```

Compile takes ~20-25 min on 4 cores. Also needs `nvidia-cuda-runtime-cu11`
installed in the venv (`pip install nvidia-cuda-runtime-cu11==11.7.99`) plus
its lib dir on `LD_LIBRARY_PATH` at *runtime* -- the compiled `.so` links
`libcudart.so.11.0`, which cu128's torch wheels don't ship:

```bash
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

Copy the resulting `.whl` from `/tmp/mmcv_wheel_out/` here (it's node-local
scratch, gone once the job ends). If the Python/arch tag changed, update the
filename in the `uv pip install --force-reinstall` command above (step 4)
and in this file's own filename references.
