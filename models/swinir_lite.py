"""
SwinIR-lite: compact Swin Transformer super-resolution backbone.

Reference architecture: Liang et al., "SwinIR: Image Restoration Using Swin
Transformer" (2021), classical-SR head variant. This is a from-scratch,
dependency-free reimplementation sized down per the dev plan (Phase 1):
RSTB depth 4, embed dim 60, window 8 -- small enough to smoke-test forward
passes on CPU, sized up (more/larger RSTBs) once real training moves to
Kaggle GPUs (see train/config/*.yaml).

Used for:
- Track B (Phase 1): 20m -> 10m, real ground truth, scale=2
- Track A (Phase 2): 10m -> 2.5m, Wald-protocol synthetic degradation, scale=4
Same architecture, different `scale` and `in_chans`/`out_chans` config.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window attention primitives
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window, with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1)*(2*Ww-1) x num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C), N = window_size[0]*window_size[1]"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, num_heads, N, N

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(N, N, -1).permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    """One Swin block: (shifted) window attention + MLP, both with residual + LayerNorm."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def _attn_mask(self, H, W, device):
        """Mask so shifted windows don't attend across the wrap-around boundary."""
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        mask = self._attn_mask(H, W, x.device)
        attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: `depth` Swin blocks (alternating shift) + conv, residual."""

    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        H, W = x_size
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size)
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + shortcut


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler; scale must be a power of 2 (2 or 4 here)."""

    def __init__(self, scale, num_feat):
        assert scale in (2, 4), "Upsample supports scale 2 or 4"
        layers = []
        for _ in range(int(math.log2(scale))):
            layers += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        super().__init__(*layers)


class SwinIRLite(nn.Module):
    """
    Compact SwinIR for satellite band super-resolution.

    in_chans/out_chans: number of spectral bands processed together.
    scale: 2 (Track B, 20m->10m) or 4 (Track A, 10m->2.5m).
    """

    def __init__(self, in_chans=4, out_chans=None, scale=2, embed_dim=60,
                 depths=(2, 2, 2, 2), num_heads=6, window_size=8, mlp_ratio=2.0):
        super().__init__()
        out_chans = out_chans or in_chans
        self.window_size = window_size
        self.scale = scale

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.layers = nn.ModuleList([
            RSTB(embed_dim, depth, num_heads, window_size, mlp_ratio) for depth in depths
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

    def _check_image_size(self, x):
        """Pad H,W up to a multiple of window_size (required for window attention)."""
        _, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), pad_h, pad_w

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        for layer in self.layers:
            x = layer(x, (H, W))
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        x, pad_h, pad_w = self._check_image_size(x)

        feat = self.conv_first(x)
        feat = feat + self.conv_after_body(self.forward_features(feat))
        feat = self.conv_before_upsample(feat)
        feat = self.upsample(feat)
        out = self.conv_last(feat)

        # Crop back the reflect-padding (scaled by `scale`).
        out = out[:, :, : orig_h * self.scale, : orig_w * self.scale]
        return out


if __name__ == "__main__":
    for scale, in_chans in [(2, 4), (4, 10)]:
        model = SwinIRLite(in_chans=in_chans, scale=scale)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.rand(1, in_chans, 32, 32)
        with torch.no_grad():
            y = model(x)
        print(f"scale={scale} in_chans={in_chans}: {tuple(x.shape)} -> {tuple(y.shape)} "
              f"({n_params/1e6:.2f}M params)")
        assert y.shape == (1, in_chans, 32 * scale, 32 * scale)
    print("Shape checks passed.")
