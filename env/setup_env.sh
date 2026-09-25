#!/bin/bash
# Source this at the start of EVERY new shell/job (after env/build_env.sh has run once):
#   source natural_language_autoencoders/env/setup_env.sh
# Paths are derived from this repo's location, so it works wherever the repo is cloned.

_NLA_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# --- paths ---
export NLA_DIR=$_NLA_REPO
export NLA_ROOT=${NLA_ROOT:-$(dirname "$_NLA_REPO")}
export BASE_DIR=${BASE_DIR:-$NLA_ROOT/miles_build_v2}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$NLA_ROOT/microtools/root}

# --- micromamba (the binary lives in $HOME, so it is lost on job restarts) ---
export PATH=~/.local/bin:$PATH
if ! command -v micromamba &>/dev/null; then
  echo "micromamba missing — reinstalling to ~/.local/bin"
  mkdir -p ~/.local/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/.local bin/micromamba
fi
eval "$(micromamba shell hook --shell bash)" 2>/dev/null

if [ ! -d "$MAMBA_ROOT_PREFIX/envs/miles" ]; then
  echo "env not found at $MAMBA_ROOT_PREFIX/envs/miles — build it first:"
  echo "  bash $_NLA_REPO/env/build_env.sh"
  return 1 2>/dev/null || exit 1
fi
micromamba activate "$MAMBA_ROOT_PREFIX/envs/miles"
export CUDA_HOME="$CONDA_PREFIX"

echo "env: $(which python)"
python -c "import torch,sglang,transformers; print('torch',torch.__version__,'sglang',sglang.__version__,'transformers',transformers.__version__)" 2>/dev/null \
  || echo "env not importable — may need rebuild"

# flashinfer JIT needs to link against the CUDA driver stub (fixes "cannot find -lcuda")
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

# sanity: is the sglang b64 transport patch still in place?
grep -q "input_embeds_b64_bf16" "$BASE_DIR/sglang/python/sglang/srt/entrypoints/http_server.py" 2>/dev/null \
  && echo "sglang NLA patch: OK" \
  || echo "WARNING: sglang NLA transport patch MISSING — rerun env/build_env.sh"
