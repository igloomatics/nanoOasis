"""Validate generated data, VAE reconstruction and matching DiT tensor shapes."""
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from data import EpisodeWindowDataset
from vae import VAE
from model import DiT
from diffusion import EDMDiffusion
from train import encode_window

p = argparse.ArgumentParser()
p.add_argument('--config', default='breakout')
p.add_argument('--vae', default=None)
a = p.parse_args()
cfg = OmegaConf.load(f'configs/{a.config}.yaml')
vae = VAE(cfg.vae)
if a.vae:
    vae.load_state_dict(torch.load(a.vae, map_location='cpu', weights_only=False)['model'])
vae.eval()
for split in ['train', 'val']:
    ds = EpisodeWindowDataset(cfg.data.index_path, split=split)
    frames, actions = next(iter(ds))
    assert frames.shape == (17, 192, 192, 3)
    assert np.all(actions < 3)
    print(split, 'shards:', len(ds.episodes), 'frame shape:', frames.shape)
x = torch.from_numpy(frames).unsqueeze(0)
with torch.no_grad():
    z = encode_window(vae, x)
    if a.vae:
        mu, _ = vae.encode(x[0])
        recon = vae.decode(mu)
        target = x[0].float() / 127.5 - 1
        print('validation L1:', (recon-target).abs().mean().item())
        import imageio.v3 as iio
        rgb = ((recon.clamp(-1,1)+1)*127.5).byte().numpy()
        iio.imwrite('breakout_reconstruction.png', np.concatenate([frames[0], rgb[0]], axis=1))
        sigma = z.std().item()
        print('latent sigma_data:', sigma, '; suggested p_mean:', np.log(1.34*sigma))
        print('suggested context_noise_max:', 0.5*sigma)
model = DiT(cfg.dit)
diff = EDMDiffusion(model, cfg.diffusion)
t = cfg.dit.context_frames + cfg.diffusion.ar_steps
loss, _ = diff.loss(z[:, -t:], torch.from_numpy(actions[-t:]).unsqueeze(0).long())
loss.backward()
assert torch.isfinite(loss)
print('DiT loss/backward OK:', loss.item())
