"""
Dual-head SRM (Super Resolution Mapping) model: shared SwinIR-lite encoder,
one head for spectral SR, one head for sub-pixel land-cover class fractions.

This is the direct response to the actual NTRO problem-statement text (see
FOUR_DAY_PLAN.md section 0.1): "converting medium-resolution multi-spectral
bands into high-resolution land cover maps ... while solving the mixed pixel
problem." The SR head alone (models/swinir_lite.py, used by Track B) only
sharpens reflectance -- it never answers the mixed-pixel / land-cover part
of the ask. The SRM head does: instead of a single hard class per output
pixel, it predicts a per-class abundance FRACTION vector, softmax-normalized
so it always sums to 1 -- which is the classical sub-pixel mapping solution
(a coarse pixel that is 60% cropland / 40% built-up comes out as [0.6, 0.4,
...] at the fine grid, not a single guessed label).

Building blocks (RSTB, Upsample) are imported from swinir_lite.py rather
than reimplemented, so both heads share one encoder and one set of Swin
weights -- SwinIRLite itself is left untouched (Track B keeps using it as-is).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swinir_lite import RSTB, Upsample


class SwinIRSRM(nn.Module):
    """
    in_chans: input spectral bands (10 for the full S2 stack).
    n_classes: land-cover classes (11 for ESA WorldCover, see
               data/worldcover_fetch.py CLASS_NAMES).
    scale: spatial upscale factor, shared by both heads (2 or 4).
    """

    def __init__(self, in_chans=10, n_classes=11, scale=4, embed_dim=60,
                 depths=(2, 2, 2, 2), num_heads=6, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.scale = scale
        self.n_classes = n_classes

        # --- shared encoder (identical pattern to SwinIRLite) ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, depth, num_heads, window_size, mlp_ratio) for depth in depths
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- SR head: reflectance, same output band count as input ---
        self.sr_pre = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.sr_upsample = Upsample(scale, embed_dim)
        self.sr_out = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        # --- SRM head: per-class abundance fractions ---
        self.srm_pre = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.srm_upsample = Upsample(scale, embed_dim)
        self.srm_out = nn.Conv2d(embed_dim, n_classes, 3, 1, 1)

    def _check_image_size(self, x):
        _, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), pad_h, pad_w

    def _encode(self, x):
        B, C, H, W = x.shape
        feat = self.conv_first(x)
        tokens = feat.flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens, (H, W))
        tokens = self.norm(tokens)
        feat_out = tokens.transpose(1, 2).view(B, feat.shape[1], H, W)
        return feat + self.conv_after_body(feat_out)

    def forward(self, x):
        """
        Returns:
            sr: (B, in_chans, H*scale, W*scale) -- reflectance
            fractions: (B, n_classes, H*scale, W*scale) -- softmax over
                       classes at every fine pixel, sums to 1 by construction
        """
        orig_h, orig_w = x.shape[-2:]
        x_pad, pad_h, pad_w = self._check_image_size(x)

        shared = self._encode(x_pad)

        sr_feat = self.sr_upsample(self.sr_pre(shared))
        sr = self.sr_out(sr_feat)
        sr = sr[:, :, : orig_h * self.scale, : orig_w * self.scale]

        srm_feat = self.srm_upsample(self.srm_pre(shared))
        logits = self.srm_out(srm_feat)
        logits = logits[:, :, : orig_h * self.scale, : orig_w * self.scale]
        fractions = torch.softmax(logits, dim=1)

        return sr, fractions


if __name__ == "__main__":
    model = SwinIRSRM(in_chans=10, n_classes=11, scale=4, embed_dim=60,
                       depths=(2, 2), num_heads=6, window_size=8)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.rand(1, 10, 32, 32)
    with torch.no_grad():
        sr, fractions = model(x)
    print(f"input {tuple(x.shape)} -> sr {tuple(sr.shape)}, fractions {tuple(fractions.shape)} "
          f"({n_params/1e6:.2f}M params)")
    assert sr.shape == (1, 10, 128, 128)
    assert fractions.shape == (1, 11, 128, 128)
    assert torch.allclose(fractions.sum(dim=1), torch.ones(1, 128, 128), atol=1e-5), \
        "fractions must sum to 1 at every pixel"
    print("Shape + fraction-sum checks passed.")
