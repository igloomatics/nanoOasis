"""Breakout rollout; no Snake referee and no rule-generated rollout frames."""
import argparse
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from game import Game, NONE
from vae import VAE
from model import DiT
from diffusion import EDMDiffusion
def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def edm_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float,
                       rho: float = 7.0, device: str = "cpu") -> torch.Tensor:
    # Karras EDM Eq. 5 -- exponentially-ramped schedule from sigma_max down to sigma_min
    ramp = torch.linspace(0, 1, num_steps + 1, device=device)
    return (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def make_schedule(cfg, num_steps: int, sigma_max: float, device: str) -> torch.Tensor:
    # sampling sigma_max decoupled from the training sigma_max 80 (D029): DIAMOND samples from
    # 10x sigma_data (5.0 with sigma_data 0.5); 0 = use that default.
    smax = sigma_max if sigma_max > 0 else 10.0 * float(cfg.diffusion.sigma_data)
    return edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, smax, device=device)


@torch.no_grad()
def sample_next_frame(diff: EDMDiffusion, z_ctx: torch.Tensor,
                      full_actions: torch.Tensor, sigmas: torch.Tensor,
                      sigma_stab: float = 0.1, sampler: str = "euler",
                      cfg_scale: float | None = None) -> torch.Tensor:
    # z_ctx: (B=1, T-1, C, Hp, Wp) clean context; full_actions: (T,) ints incl. the new action.
    # sampler: "euler" = 1st-order deterministic, the DIAMOND-proven few-step regime (D029);
    #          "heun" = Karras 2nd-order, quality fallback at >= 8 steps.
    # cfg_scale: classifier-free guidance on ACTION (the trained null row, D029); >1 amplifies the
    # action's pull when drift drowns control. Default off; set via NANO_CFG=1.5 etc.
    if cfg_scale is None:
        cfg_scale = float(os.environ.get("NANO_CFG", "0"))
    B = z_ctx.shape[0]
    T = z_ctx.shape[1] + 1
    C, Hp, Wp = z_ctx.shape[2:]
    device = z_ctx.device

    if full_actions.dim() == 1:
        full_actions = full_actions.unsqueeze(0)

    noisy_ctx = z_ctx + sigma_stab * torch.randn_like(z_ctx)
    x_new = torch.randn(B, 1, C, Hp, Wp, device=device) * sigmas[0]
    null_actions = torch.full_like(full_actions, diff.model.num_actions)

    def deriv(x, sigma):                                       # score-like estimate at the new frame
        sig = torch.cat([torch.full((B, T - 1), sigma_stab, device=device),
                         torch.full((B, 1),     sigma,      device=device)], dim=1)
        x_in = torch.cat([noisy_ctx, x], dim=1)
        D = diff.denoise(x_in, sig, full_actions)
        if cfg_scale > 0:
            D_null = diff.denoise(x_in, sig, null_actions)     # whole-window null matches the dropout regime
            D = D_null + cfg_scale * (D - D_null)
        return (x - D[:, -1:]) / sigma

    for i in range(len(sigmas) - 1):
        sigma_cur = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        d = deriv(x_new, sigma_cur)
        x_euler = x_new + (sigma_next - sigma_cur) * d
        if sampler == "heun" and sigma_next > 0:               # Karras EDM Alg. 1 -- 2nd-order correction
            d2 = deriv(x_euler, sigma_next)
            x_new = x_new + (sigma_next - sigma_cur) * 0.5 * (d + d2)
        else:
            x_new = x_euler
    if device.type == "mps":
        torch.mps.empty_cache()                                # free per-frame so long rollouts don't OOM
    return x_new


@torch.no_grad()
def decode_latent(vae: VAE, z: torch.Tensor) -> np.ndarray:
    # z: (1, 1, C, Hp, Wp) -> (H, W, 3) uint8
    z_tok = z.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, vae.latent_channels).unsqueeze(0)
    recon = vae.decode(z_tok)
    return ((recon.squeeze(0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()



@torch.no_grad()
def main(args):
    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    device = pick_device("auto")
    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(torch.load(args.vae, map_location=device, weights_only=False)["model"])
    model = DiT(cfg.dit).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if args.raw else ck.get("ema", ck["model"]))
    diff = EDMDiffusion(model, cfg.diffusion).to(device).eval()
    from data_gen import SeekBot
    g = Game(args.seed)
    bot = SeekBot(np.random.default_rng(args.seed))
    frames, actions = [], []
    for _ in range(cfg.dit.context_frames):
        a = bot.act(g); frames.append(g.step(a)[0]); actions.append(a)
    mu, _ = vae.encode(torch.from_numpy(np.stack(frames)).to(device))
    z = mu.reshape(1, len(frames), vae.Hp, vae.Wp, vae.latent_channels).permute(0,1,4,2,3).contiguous()
    schedule = make_schedule(cfg, args.steps, 0, device)
    output = []
    for t in range(args.frames):
        # Fixed independent actions; the real game is never advanced during rollout.
        a = (t // 12) % 3
        actions = actions[1:] + [a]
        new = sample_next_frame(diff, z[:, 1:], torch.tensor(actions, device=device), schedule)
        z = torch.cat([z[:, 1:], new], dim=1)
        output.append(decode_latent(vae, new))
    import imageio.v3 as iio
    iio.imwrite(args.out, np.stack(output), duration=1000/24, loop=0)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="breakout_latents")
    p.add_argument("--vae", default="checkpoints/vae_breakout.pt")
    p.add_argument("--ckpt", default="checkpoints/dit_breakout_latents.pt")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw", action="store_true", help="use raw weights for short smoke runs")
    p.add_argument("--out", default="breakout_rollout.gif")
    main(p.parse_args())
