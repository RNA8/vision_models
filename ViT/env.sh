# Keep every tool cache inside the repo (gitignored .cache/), not in ~/.cache, ~/.triton or /tmp.
# Sourced by the run_*.sh scripts; for an interactive shell: `source ViT/env.sh`.
# (wandb's login credential stays in ~/.netrc, where `wandb login` put it.)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_CACHE="$REPO/.cache"
export PIP_CACHE_DIR="$REPO_CACHE/pip"
export HF_HOME="$REPO_CACHE/huggingface"           # `hf auth login` token lands here too
export TORCHINDUCTOR_CACHE_DIR="$REPO_CACHE/torchinductor"
export TRITON_CACHE_DIR="$REPO_CACHE/triton"
export WANDB_CACHE_DIR="$REPO_CACHE/wandb"
export WANDB_CONFIG_DIR="$REPO_CACHE/wandb-config"
export WANDB_DATA_DIR="$REPO_CACHE/wandb-data"
export MPLCONFIGDIR="$REPO_CACHE/matplotlib"
export XDG_CACHE_HOME="$REPO_CACHE"
export TMPDIR="$REPO_CACHE/tmp"
mkdir -p "$REPO_CACHE/tmp"
