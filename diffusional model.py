"""
Denoising Diffusion Probabilistic Model (DDPM) — PyTorch
=========================================================
Paper: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

Architecture:
  - GaussianDiffusion : forward/reverse process + loss
  - UNet              : noise-prediction backbone (time-conditioned)
  - SinusoidalPE      : sinusoidal time embeddings
  - ResBlock          : residual block with group-norm + time injection
  - AttentionBlock    : self-attention for spatial features
  - Trainer           : training loop with EMA + checkpoint saving
"""

import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


# ─────────────────────────────────────────────────────────────────────────────
# 1. NOISE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal 2021).
    Produces smoother noise than the linear schedule.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]          # normalise to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02) -> Tensor:
    """Original linear schedule from Ho et al. 2020."""
    return torch.linspace(beta_start, beta_end, T)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN DIFFUSION  (forward + reverse + loss)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    Encapsulates all maths for DDPM.

    Forward (noising):
        q(x_t | x_0) = N(x_t ; sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I)

    Reverse (denoising):
        p_θ(x_{t-1} | x_t) = N(x_{t-1} ; μ_θ(x_t, t), σ_t²·I)
        where μ_θ reconstructs x_0 by subtracting the predicted noise.

    Loss:
        L = E_{x_0, ε, t} [ ||ε - ε_θ(x_t, t)||² ]
    """

    def __init__(self, model: nn.Module, T: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.model = model
        self.T = T

        # Build noise schedule and derived quantities
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T)

        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)         # ᾱ_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers so they move with .to(device) and are saved in state_dict
        self.register_buffer("betas",                betas)
        self.register_buffer("alphas_cumprod",       alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",  alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",  alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

        # Quantities for the reverse mean formula
        self.register_buffer("sqrt_recip_alphas", alphas.rsqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract(a: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather values from schedule tensor `a` at timesteps `t`, reshape to `shape`."""
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    # ── forward process ──────────────────────────────────────────────────────

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Sample from q(x_t | x_0) using the reparameterisation trick.
        x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar   = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus   = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    # ── loss ─────────────────────────────────────────────────────────────────

    def p_losses(self, x_start: Tensor, t: Tensor) -> Tensor:
        """
        Compute the simplified DDPM loss: E[||ε - ε_θ(x_t, t)||²].
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(noise, predicted_noise)

    def forward(self, x: Tensor) -> Tensor:
        """Sample a random timestep and return the loss."""
        b, *_ = x.shape
        t = torch.randint(0, self.T, (b,), device=x.device)
        return self.p_losses(x, t)

    # ── reverse process (sampling) ────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor, t_index: int) -> Tensor:
        """
        One step of the reverse process:
        x_{t-1} ~ p_θ(x_{t-1} | x_t)
        """
        betas_t           = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_t  = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_t      = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Predict noise, then compute model mean μ_θ
        predicted_noise = self.model(x, t)
        model_mean = sqrt_recip_t * (x - betas_t * predicted_noise / sqrt_one_minus_t)

        if t_index == 0:
            return model_mean                   # no noise at final step
        else:
            posterior_variance = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + posterior_variance.sqrt() * noise

    @torch.no_grad()
    def sample(self, image_size: int, batch_size: int = 16, channels: int = 3) -> Tensor:
        """
        Full reverse diffusion: start from pure noise, denoise T steps.
        Returns x_0 samples in [-1, 1].
        """
        device = next(self.model.parameters()).device
        shape  = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=device)

        for t_index in reversed(range(self.T)):
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_index)

        return x.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. UNet BACKBONE
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional encoding for timestep t.
    Encodes the diffusion timestep into a fixed-dimension embedding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device   = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


class ResBlock(nn.Module):
    """
    Residual block with:
      • GroupNorm + SiLU activation
      • Time embedding injected via a linear projection
      • Optional residual projection if channels change
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]   # broadcast spatial dims
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Single-head spatial self-attention with pre-norm (GroupNorm).
    Applied at the bottleneck (and optionally at lower resolutions).
    """
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm  = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj  = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(dim=1)                        # each (B, C, HW)

        scale   = C ** -0.5
        attn    = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)  # (B, HW, HW)
        out     = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    Time-conditioned UNet for noise prediction.

    Encoder path  : progressively downsamples the feature map.
    Bottleneck    : ResBlock + Attention + ResBlock at the coarsest scale.
    Decoder path  : upsamples with skip connections from encoder.

    Args:
        image_channels : number of input/output channels (1 = grayscale, 3 = RGB)
        base_channels  : base feature-map width (doubled at each level)
        channel_mults  : multipliers per resolution level
        T              : total diffusion timesteps (for embedding size)
        attn_levels    : which encoder levels get self-attention
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels:  int = 64,
        channel_mults:  tuple[int, ...] = (1, 2, 4, 8),
        T:              int = 1000,
        attn_levels:    tuple[int, ...] = (2,),    # add attention at 3rd level (0-indexed)
        groups:         int = 8,
    ):
        super().__init__()
        time_emb_dim = base_channels * 4

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPE(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Initial projection ─────────────────────────────────────────────────
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.downs     = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = base_channels
        for level, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            use_attn = (level in attn_levels)

            block = nn.ModuleList([
                ResBlock(ch_in, ch_out, time_emb_dim, groups),
                ResBlock(ch_out, ch_out, time_emb_dim, groups),
                AttentionBlock(ch_out, groups) if use_attn else nn.Identity(),
            ])
            self.downs.append(block)

            # Downsample (stride-2 conv) except at the final level
            if level < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch_out, ch_out, 4, 2, 1))
            else:
                self.downsamples.append(nn.Identity())
            ch_in = ch_out

        # ── Bottleneck ─────────────────────────────────────────────────────────
        mid_ch = base_channels * channel_mults[-1]
        self.mid_block1  = ResBlock(mid_ch, mid_ch, time_emb_dim, groups)
        self.mid_attn    = AttentionBlock(mid_ch, groups)
        self.mid_block2  = ResBlock(mid_ch, mid_ch, time_emb_dim, groups)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.ups       = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_mults = list(reversed(channel_mults))
        for level, mult in enumerate(rev_mults):
            ch_skip = base_channels * mult
            ch_out  = base_channels * rev_mults[min(level + 1, len(rev_mults) - 1)]
            use_attn = ((len(channel_mults) - 1 - level) in attn_levels)

            block = nn.ModuleList([
                ResBlock(ch_in + ch_skip, ch_in, time_emb_dim, groups),
                ResBlock(ch_in, ch_in, time_emb_dim, groups),
                AttentionBlock(ch_in, groups) if use_attn else nn.Identity(),
            ])
            self.ups.append(block)

            if level < len(rev_mults) - 1:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(ch_in, ch_out, 3, padding=1),
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            ch_in = ch_out

        # ── Final output head ─────────────────────────────────────────────────
        self.out = nn.Sequential(
            nn.GroupNorm(groups, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x : noisy image  (B, C, H, W)  in [-1, 1]
            t : timesteps    (B,)           integers in [0, T)
        Returns:
            predicted noise  (B, C, H, W)
        """
        time_emb = self.time_mlp(t)
        x = self.init_conv(x)

        # Encoder — store skip connections
        skips = []
        for (r1, r2, attn), down in zip(self.downs, self.downsamples):
            x = r1(x, time_emb)
            x = r2(x, time_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Decoder — concatenate skip connections
        for (r1, r2, attn), up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = r1(x, time_emb)
            x = r2(x, time_emb)
            x = attn(x)
            x = up(x)

        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMA  (Exponential Moving Average of model weights)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Maintains a shadow copy of model weights updated as a running mean.
    Using EMA weights for inference significantly improves sample quality.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.shadow = copy.deepcopy(model).eval()
        self.decay  = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Training loop for DDPM.

    Features:
      • Mixed-precision (bfloat16 / float16) via torch.amp
      • Gradient clipping
      • EMA weight tracking
      • Periodic checkpoint + sample saving
    """

    def __init__(
        self,
        diffusion:   GaussianDiffusion,
        dataloader:  DataLoader,
        lr:          float = 1e-4,
        grad_clip:   float = 1.0,
        ema_decay:   float = 0.9999,
        save_every:  int   = 1000,
        sample_every:int   = 500,
        results_dir: str   = "results",
        image_size:  int   = 64,
        amp:         bool  = True,
    ):
        self.diffusion    = diffusion
        self.dataloader   = dataloader
        self.optimizer    = torch.optim.AdamW(diffusion.model.parameters(), lr=lr)
        self.ema          = EMA(diffusion.model, decay=ema_decay)
        self.grad_clip    = grad_clip
        self.save_every   = save_every
        self.sample_every = sample_every
        self.results_dir  = Path(results_dir)
        self.image_size   = image_size
        self.scaler       = torch.amp.GradScaler(enabled=amp)
        self.amp          = amp
        self.step         = 0
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train(self, total_steps: int):
        device = next(self.diffusion.model.parameters()).device
        data_iter = iter(self.dataloader)

        self.diffusion.model.train()
        for _ in range(total_steps):
            # Refresh iterator when exhausted
            try:
                batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch, _ = next(data_iter)

            batch = batch.to(device)

            # Forward + loss
            with torch.amp.autocast(device_type=device.type, enabled=self.amp):
                loss = self.diffusion(batch)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.diffusion.model)

            self.step += 1
            if self.step % 100 == 0:
                print(f"step {self.step:6d} | loss {loss.item():.4f}")

            # Save samples
            if self.step % self.sample_every == 0:
                self._save_samples()

            # Save checkpoint
            if self.step % self.save_every == 0:
                self._save_checkpoint()

    @torch.no_grad()
    def _save_samples(self):
        self.ema.shadow.eval()
        # Temporarily swap to EMA weights for sampling
        tmp_model = self.diffusion.model
        self.diffusion.model = self.ema.shadow

        samples = self.diffusion.sample(
            image_size=self.image_size, batch_size=16,
            channels=next(self.ema.shadow.parameters()).shape[0]
        )
        samples = (samples + 1) / 2           # [-1,1] → [0,1]
        utils.save_image(
            samples, self.results_dir / f"sample_{self.step:06d}.png", nrow=4
        )
        self.diffusion.model = tmp_model

    def _save_checkpoint(self):
        ckpt = {
            "step":       self.step,
            "model":      self.diffusion.model.state_dict(),
            "ema_shadow": self.ema.shadow.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }
        torch.save(ckpt, self.results_dir / f"ckpt_{self.step:06d}.pt")
        print(f"  ↳ checkpoint saved at step {self.step}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN  (example: train on CIFAR-10)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    IMAGE_SIZE   = 32
    BATCH_SIZE   = 128
    TOTAL_STEPS  = 100_000
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
    dataset    = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    unet = UNet(
        image_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        T=1000,
        attn_levels=(2,),           # attention at 4× downsampled features
    ).to(DEVICE)

    diffusion = GaussianDiffusion(model=unet, T=1000, schedule="cosine").to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        diffusion=diffusion,
        dataloader=dataloader,
        lr=2e-4,
        ema_decay=0.9999,
        image_size=IMAGE_SIZE,
        save_every=5000,
        sample_every=1000,
        results_dir="results/cifar10",
        amp=True,
    )
    trainer.train(TOTAL_STEPS)


if __name__ == "__main__":
    main()