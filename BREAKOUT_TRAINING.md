# Breakout training

Run commands from the repository root, using its virtual environment.

```bash
source .venv/bin/activate
python -m pip install -e .
python data_gen.py --n-frames 10000 --workers 4 --out data/breakout
python validate_breakout.py
python train_vae.py --config breakout --steps 500
python validate_breakout.py --vae checkpoints/vae_breakout.pt
```

Inspect `breakout_reconstruction.png`: the ball must remain visible and bricks must reconstruct sharply. Before longer DiT training, update `sigma_data`, `p_mean` and `context_noise_max` in `configs/breakout_latents.yaml` using the values printed by validation. These initial configuration values are placeholders, not measurements.

```bash
python encode_latents.py --index data/breakout/index.parquet --vae checkpoints/vae_breakout.pt --config breakout --out data/breakout_latents
python train.py --config breakout_latents --steps 500
python infer_breakout.py --raw --out breakout_rollout.gif
```

The 500-step runs only check the pipeline; they do not promise a playable model. Once reconstruction and short rollouts are good, increase data and training steps. Keep a frozen VAE while training DiT; regenerate latent shards whenever VAE weights change. Use `breakout` for raw-frame VAE training and `breakout_latents` for DiT training. They have separate checkpoint names from Snake.

Data contains post-action RGB frames, aligned actions, event markers (brick hits, paddle bounces and life losses), and cumulative life-loss markers. Worker 0 is validation; other workers are training. Each shard uses a separate seed. The generator mixes 80% noisy tracking bots with 20% random bots. Fixed-size shards bound memory. The new rollout script generates frames exclusively through DiT after a short real-game seed context.

The old Snake demo, ONNX files, `infer.py` and tests are not compatible with Breakout. Use `infer_breakout.py`; browser export/adaptation is a separate step. The original Snake smoke dataset must not be used for Breakout.
