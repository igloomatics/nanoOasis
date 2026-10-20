#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python play_breakout_model.py \
  --config breakout_100k_latents \
  --vae checkpoints/vae_breakout_100k.pt \
  --ckpt checkpoints/dit_breakout_100k_latents.pt \
  --steps "${1:-4}" \
  --fps "${2:-4}"
