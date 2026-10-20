#!/usr/bin/env bash
set -euo pipefail
cd /root/nanoOasis
PY=.venv/bin/python
TORCHRUN="$PY -m torch.distributed.run --standalone --nproc_per_node=4"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo '[1/4] Four-GPU VAE training'
$TORCHRUN train_vae.py --config breakout_500k --steps 50000

echo '[2/4] Measure VAE and encode 500k latents'
$PY validate_breakout.py --config breakout_500k --vae checkpoints/vae_breakout_500k.pt | tee /tmp/validate_breakout_500k.log
SIGMA=$($PY - <<'PY'
import re
s=open('/tmp/validate_breakout_500k.log').read(); print(re.search(r'latent sigma_data: ([0-9.e+-]+)',s).group(1))
PY
)
PMEAN=$($PY - <<PY
import math
print(math.log(1.34*float('$SIGMA')))
PY
)
CNMAX=$($PY - <<PY
print(0.5*float('$SIGMA'))
PY
)
$PY encode_latents.py --index data/breakout_500k/index.parquet --vae checkpoints/vae_breakout_500k.pt --config breakout_500k --out data/breakout_500k_latents
cp configs/breakout_500k.yaml configs/breakout_500k_latents.yaml
sed -i -e 's#data/breakout_500k/index.parquet#data/breakout_500k_latents/index.parquet#' -e 's/sigma_data: 0.5/sigma_data: '$SIGMA'/' -e 's/p_mean: -0.4/p_mean: '$PMEAN'/' -e 's/context_noise_max: 0.25/context_noise_max: '$CNMAX'/' configs/breakout_500k_latents.yaml
sed -i '/^data:/a\  pre_encoded: true' configs/breakout_500k_latents.yaml

echo '[3/4] Four-GPU DiT training'
$TORCHRUN train.py --config breakout_500k_latents --steps 200000

echo '[4/4] Start one-GPU browser inference service'
export CUDA_VISIBLE_DEVICES=0
exec env SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy XDG_RUNTIME_DIR=/tmp $PY -u web_play_breakout.py --config breakout_500k_latents --vae checkpoints/vae_breakout_500k.pt --ckpt checkpoints/dit_breakout_500k_latents.pt --port 8765 --steps 2
