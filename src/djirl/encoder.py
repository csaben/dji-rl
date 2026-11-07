"""Frozen image encoder: frames -> compact latent (comma.ai-style world/latent input).

Default is TAESD's encoder (madebyollin/taesd, ~1M params, 8x downsample, 4 channels):
128x96 RGB -> 4x12x16 latent. Swap for a VAE fine-tuned on sim+real frames later."""
from __future__ import annotations

import torch


class LatentEncoder(torch.nn.Module):
    def __init__(self, device, name: str = "madebyollin/taesd", dtype=torch.float16):
        super().__init__()
        import safetensors.torch  # noqa: F401  (diffusers 0.31 expects it pre-imported)
        from diffusers import AutoencoderTiny
        self.enc = AutoencoderTiny.from_pretrained(name, torch_dtype=dtype).encoder.to(device).eval()
        self.enc.requires_grad_(False)
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb [B,H,W,3] in [0,1] -> latent [B,4,H/8,W/8] float32."""
        x = rgb[..., :3].permute(0, 3, 1, 2).to(self.dtype) * 2 - 1
        return self.enc(x).float()
