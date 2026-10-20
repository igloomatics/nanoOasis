# Long-running training runbook

## Always-on rule

Long-running training must never depend on a foreground Codex `exec_command` session. A tool session can be reclaimed between turns, which can terminate attached child processes without a Python traceback. Before starting multi-hour work:

1. save data and checkpoints under the project directory;
2. redirect stdout/stderr to a persistent log;
3. detach the process with `setsid`/`nohup` or a scheduler;
4. record the PID and launch command;
5. verify the process, GPU utilization, log growth, and checkpoint mtime from a separate command;
6. save optimizer state and the current step so interruption can resume.

Never report a long run as healthy based only on the fact that the launch command returned a session ID.

## Current 500k pipeline

`run_500k_4gpu_then_web.sh` runs four GPU VAE training, latent encoding, four GPU DiT training, and finally one GPU browser inference. Launch it detached and inspect `/root/nanoOasis/500k-training.log` plus checkpoint steps.

## Current run

- Launch command: `setsid bash -c './run_500k_4gpu_then_web.sh >500k-training.log 2>&1' </dev/null >/dev/null 2>&1 &`
- PID file: `/root/nanoOasis/500k-training.pid`
- Log: `/root/nanoOasis/500k-training.log`
- Pipeline: 500k frames, 4-GPU VAE, latent encoding, 4-GPU DiT (200k steps), then 1-GPU web service.

## Incident lesson

The earlier run was started in a foreground tool session. When that session was reclaimed, the child training processes disappeared without a useful Python traceback. From now on, a foreground session is used only for smoke tests; every multi-hour run is detached and must be checked through its persistent PID, log, checkpoint step, and GPU utilization.

## Breakout 500k retraining configuration

### Dataset

- Game: custom Python/Pygame Breakout; no Atari environment or ROM.
- Frames: `500,000`.
- Dataset directory: `data/breakout_500k`.
- Latent dataset directory: `data/breakout_500k_latents`.
- Data generation used multiple workers and retained event-rich windows.

### VAE

- Training: `50,000` steps.
- Hardware: 4 NVIDIA GPUs with DDP.
- Checkpoint: `checkpoints/vae_breakout_500k.pt`.
- Final validation L1: approximately `0.0019` on the training log.
- The VAE is frozen during DiT training.

### Latent encoding

- Encoder: the completed VAE encoder mean (`mu`), with no sampling noise.
- Latent normalization: measured `sigma_data = 1.327248215675354`.
- The 500k frames were encoded into `252` latent shards.

### DiT

- Training: `200,000` steps.
- Hardware: 4 NVIDIA GPUs with DDP.
- Context length: `8` frames.
- Training windows use one target frame plus the 8-frame context; DDP training uses a single backward pass per window for stability.
- EDM parameters: `sigma_data=1.327248215675354`, `p_mean=0.5757774020663547`, `p_std=1.0`, `sigma_min=0.002`, `sigma_max=80.0`.
- Batch size per GPU: `4`.
- Learning rate: `1e-4`, warmup `200` steps, cosine decay, gradient clipping `1.0`.
- Checkpoint: `checkpoints/dit_breakout_500k_latents.pt`.

### Inference

- After DiT reaches 200,000 steps, the pipeline starts the browser service on one GPU.
- Service command: `resume_500k_dit_then_web.sh`.
- Port: `8765`.
- Browser access through the SSH tunnel: `ssh -N -L 8765:127.0.0.1:8765 volc-sims`, then open `http://127.0.0.1:8765`.
