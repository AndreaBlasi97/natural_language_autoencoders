# Rebuilding the training env on a new machine

The micromamba env (~47 GB) and the patched sglang/miles/Megatron sources are not
in git. This folder has everything needed to rebuild them:

| File | What |
|---|---|
| `build_env.sh` | One-time build: CUDA 12.9 + torch 2.9.1 env, clones sglang/miles/Megatron-LM at pinned commits, applies patches, installs everything |
| `setup_env.sh` | Source in every new shell to activate the env |
| `requirements.lock.txt` | Exact pip versions from the working env |
| `patches/*.patch` | `git diff` of the working sglang / miles / Megatron-LM trees |

```bash
git clone https://github.com/AndreaBlasi97/natural_language_autoencoders.git
bash natural_language_autoencoders/env/build_env.sh      # once, ~1h+ (compiles flash-attn, apex, TE)
source natural_language_autoencoders/env/setup_env.sh    # every new shell
```

The env and sources go next to the clone (`../microtools`, `../miles_build_v2`);
set `NLA_ROOT` to put them elsewhere. Set `MAX_JOBS` lower if compilation runs out of RAM.
