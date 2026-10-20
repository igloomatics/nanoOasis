#!/usr/bin/env bash
set -euo pipefail
cd /root/nanoOasis
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo '[resume] Four-GPU DiT training (context=8)'
$PY -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config breakout_500k_latents --steps 200000
echo '[resume] Start one-GPU browser inference service'
export CUDA_VISIBLE_DEVICES=0
exec env SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy XDG_RUNTIME_DIR=/tmp $PY -u web_play_breakout.py --config breakout_500k_latents --vae checkpoints/vae_breakout_500k.pt --ckpt checkpoints/dit_breakout_500k_latents.pt --port 8765 --steps 2
