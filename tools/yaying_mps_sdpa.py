"""Chunked scaled_dot_product_attention for Apple MPS (torch 2.5 materialises the full attention matrix and fails with
'Invalid buffer size' on video-diffusion batches). Splits along batch*heads so each chunk's score matrix stays under
--budget bytes. Numerically identical (attention is independent per batch/head). Import and call patch() first."""
import torch
import torch.nn.functional as F

_orig = F.scaled_dot_product_attention


def _chunked(q=None, k=None, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if q is None: q = kw.pop("query")               # diffusers >=0.35 passes query=/key=/value= keywords
    if k is None: k = kw.pop("key")
    if v is None: v = kw.pop("value")
    if q.device.type != "mps" or q.dim() < 3:
        return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    L, S = q.shape[-2], k.shape[-2]
    lead = q.shape[:-2]
    nb = int(torch.tensor(lead).prod()) if len(lead) else 1
    per = L * S * q.element_size() * 2
    budget = _chunked.budget
    if nb * per <= budget:
        return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    qf, kf, vf = q.reshape(nb, 1, L, -1), k.reshape(nb, 1, S, -1), v.reshape(nb, 1, S, -1)
    mf = attn_mask.expand(*lead, L, S).reshape(nb, 1, L, S) if attn_mask is not None else None
    step = max(1, budget // per)
    outs = []
    for i in range(0, nb, step):
        if per > budget and L > 1024:                    # a single item too large: also split queries
            qs = max(256, int(budget // (S * q.element_size() * 2)))
            outs.append(torch.cat([_orig(qf[i:i + step, :, j:j + qs], kf[i:i + step], vf[i:i + step],
                                         attn_mask=None if mf is None else mf[i:i + step, :, j:j + qs], dropout_p=dropout_p,
                                         is_causal=is_causal, scale=scale) for j in range(0, L, qs)], 2))
        else:
            outs.append(_orig(qf[i:i + step], kf[i:i + step], vf[i:i + step], attn_mask=None if mf is None else mf[i:i + step],
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale))
    return torch.cat(outs, 0).reshape(*lead, L, v.shape[-1])


_chunked.budget = 2 * 1024 ** 3


def patch(budget_gb=2.0):
    _chunked.budget = int(budget_gb * 1024 ** 3)
    F.scaled_dot_product_attention = _chunked
    torch.nn.functional.scaled_dot_product_attention = _chunked
