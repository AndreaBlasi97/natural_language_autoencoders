#!/bin/bash
# Rebuild the working "miles" env (torch 2.9.1+cu129, sglang 0.5.16, transformers 5.3.0)
# from scratch on a fresh machine. Run ONCE after cloning, on a node with a GPU driver:
#
#   bash natural_language_autoencoders/env/build_env.sh
#
# Then, in every new shell:
#   source natural_language_autoencoders/env/setup_env.sh
#
# Layout (mirrors the original /work/training/NLAandrea/nla-inference setup):
#   $NLA_ROOT/natural_language_autoencoders   <- this repo
#   $NLA_ROOT/miles_build_v2/{sglang,miles,Megatron-LM}   <- patched upstream sources
#   $NLA_ROOT/microtools/root/envs/miles       <- micromamba env
# NLA_ROOT defaults to the directory containing this repo.
#
# Re-running is safe: finished steps are skipped. Compiling flash-attn / apex /
# transformer_engine is the slow part (~1h+); lower MAX_JOBS if you run out of RAM.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=$REPO_DIR/env
export NLA_ROOT=${NLA_ROOT:-$(dirname "$REPO_DIR")}
export BASE_DIR=${BASE_DIR:-$NLA_ROOT/miles_build_v2}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$NLA_ROOT/microtools/root}
ENV_PREFIX=$MAMBA_ROOT_PREFIX/envs/miles
export MAX_JOBS=${MAX_JOBS:-32}

# --- pinned commits (the set that actually works together) ---
SGLANG_COMMIT=1519acf37c23f2189adb93f57ca9cd2db1bebf18    # v0.5.10, has DumperConfig
MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
MILES_COMMIT=051cd15a2ff594759cf6c7a0bc698d1ac0e90b44
APEX_COMMIT=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
MEGATRON_BRIDGE_COMMIT=35b4ebfc486fb15dcc0273ceea804c3606be948a
TORCH_MEMORY_SAVER_COMMIT=dc6876905830430b5054325fa4211ff302169c6b

echo "NLA_ROOT=$NLA_ROOT"
echo "env      -> $ENV_PREFIX"
echo "sources  -> $BASE_DIR"

# --- 1. micromamba binary (non-interactive install into ~/.local/bin) ---
if ! command -v micromamba &>/dev/null; then
  mkdir -p ~/.local/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/.local bin/micromamba
  export PATH=~/.local/bin:$PATH
fi
eval "$(micromamba shell hook --shell bash)"

# --- 2. conda env: python + CUDA 12.9 toolkit (needed to compile the extensions) ---
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  micromamba create -p "$ENV_PREFIX" python=3.12 pip -c conda-forge -y
  micromamba install -p "$ENV_PREFIX" cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
  micromamba install -p "$ENV_PREFIX" cudnn -c conda-forge -y
fi
set +u; micromamba activate "$ENV_PREFIX"; set -u
export CUDA_HOME="$CONDA_PREFIX"
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}
PIP="python -m pip"

# --- 3. torch (cu129 wheels) ---
$PIP install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# --- 4. upstream sources at pinned commits + our patches ---
clone_and_patch() {  # <dir> <url> <commit> <patch>
  local dir=$BASE_DIR/$1
  if [ ! -d "$dir/.git" ]; then
    git clone "$2" "$dir"
    git -C "$dir" checkout "$3"
    git -C "$dir" submodule update --init --recursive
  fi
  if git -C "$dir" apply --reverse --check "$4" 2>/dev/null; then
    echo "$1: patch already applied"
  else
    git -C "$dir" apply "$4"
    echo "$1: patch applied"
  fi
}
mkdir -p "$BASE_DIR"
clone_and_patch sglang      https://github.com/sgl-project/sglang.git "$SGLANG_COMMIT"   "$ENV_DIR/patches/sglang.patch"
clone_and_patch miles       https://github.com/radixark/miles.git     "$MILES_COMMIT"    "$ENV_DIR/patches/miles.patch"
clone_and_patch Megatron-LM https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_COMMIT" "$ENV_DIR/patches/megatron.patch"

# --- 5. every PyPI package at the exact version from the working env ---
# --no-deps: the lock is a complete freeze, so skip the resolver (no version drift).
$PIP install --no-deps -r "$ENV_DIR/requirements.lock.txt"

# --- 6. packages compiled against this torch/CUDA (same recipe as miles/build_conda.sh) ---
$PIP install --no-build-isolation --no-deps flash-attn==2.7.4.post1
$PIP install --no-build-isolation --no-deps transformer_engine==2.10.0 transformer_engine_torch==2.10.0
NVCC_APPEND_FLAGS="--threads 4" $PIP install --no-build-isolation --no-deps --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  "git+https://github.com/NVIDIA/apex.git@$APEX_COMMIT"
$PIP install --no-deps --no-cache-dir "git+https://github.com/fzyzcjy/torch_memory_saver.git@$TORCH_MEMORY_SAVER_COMMIT"
$PIP install --no-deps "git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_COMMIT"
$PIP install --no-build-isolation --no-deps "git+https://github.com/fzyzcjy/Megatron-Bridge.git@$MEGATRON_BRIDGE_COMMIT"

# --- 7. editable installs (patched sources + this repo) ---
$PIP install --no-deps -e "$BASE_DIR/sglang/python"
$PIP install --no-deps -e "$BASE_DIR/Megatron-LM"
$PIP install --no-deps -e "$BASE_DIR/miles"
$PIP install --no-deps -e "$REPO_DIR"

# --- 8. verify ---
python -c "import torch,sglang,transformers,megatron.core,miles,nla; print('torch',torch.__version__,'sglang',sglang.__version__,'transformers',transformers.__version__)"
$PIP check || echo "(pip check warnings above are informational)"
echo "Done. In each new shell: source $ENV_DIR/setup_env.sh"
