"""Fast Conv3d for Apple MPS: torch 2.5 MPS conv3d is very slow for the Wan video VAE (measured 25 frames 688x464:
encode 129.7 s, decode 237.1 s fp32) and leaks graph memory. A (kt,kh,kw) conv3d with zero 3D padding equals the
sum over kt temporal taps of conv2d on time slices; conv2d is well optimised on MPS. Exact up to float rounding.
patch_echomimic() / patch_diffusers() replace the causal-conv forward of the Wan VAEs (padding is done before)."""
import torch
import torch.nn.functional as F


def conv3d_2d(x, w, b, stride=(1, 1, 1), dilation=(1, 1, 1), groups=1):
    B, C, T, H, W = x.shape
    O, Cg, kt, kh, kw = w.shape
    st, sh, sw = stride; dt, dh, dw = dilation
    To = (T - dt * (kt - 1) - 1) // st + 1
    out = None
    for i in range(kt):
        xs = x[:, :, i * dt: i * dt + st * (To - 1) + 1: st]                    # B C To H W
        xs = xs.permute(0, 2, 1, 3, 4).reshape(B * To, C, H, W)
        y = F.conv2d(xs, w[:, :, i], None, (sh, sw), 0, (dh, dw), groups)
        out = y if out is None else out + y
    if b is not None:
        out = out + b.view(1, -1, 1, 1)
    Ho, Wo = out.shape[-2:]
    return out.reshape(B, To, O, Ho, Wo).permute(0, 2, 1, 3, 4).contiguous()


def _use(x):
    return x.device.type == "mps"


def patch_echomimic():
    import src.wan_vae as wv
    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device); x = torch.cat([cache_x, x], dim=2); padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        if not _use(x):
            return torch.nn.Conv3d.forward(self, x)
        return conv3d_2d(x, self.weight, self.bias, self.stride, self.dilation, self.groups)
    wv.CausalConv3d.forward = forward


def patch_diffusers():
    from diffusers.models.autoencoders import autoencoder_kl_wan as dw
    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device); x = torch.cat([cache_x, x], dim=2); padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        if not _use(x):
            return torch.nn.Conv3d.forward(self, x)
        return conv3d_2d(x, self.weight, self.bias, self.stride, self.dilation, self.groups)
    dw.WanCausalConv3d.forward = forward


if __name__ == "__main__":
    torch.manual_seed(0)
    for (C, O, k, s) in [(8, 16, (3, 3, 3), (1, 1, 1)), (8, 8, (3, 1, 1), (2, 1, 1)), (4, 6, (1, 1, 1), (1, 1, 1))]:
        x = torch.randn(1, C, 7, 20, 18); w = torch.randn(O, C, *k); b = torch.randn(O)
        ref = F.conv3d(x, w, b, s); got = conv3d_2d(x, w, b, s)
        print(k, s, tuple(ref.shape) == tuple(got.shape), float((ref - got).abs().max()))
