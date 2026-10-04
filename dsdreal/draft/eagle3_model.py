"""EAGLE3 draft head in plain PyTorch, ported from vLLM's
model_executor/models/llama_eagle3.py (0.27.1) so it can run outside a vLLM
engine, in a standalone drafter process fed hidden states over the network.

Same math, layer for layer (vLLM names in brackets):
- fc: concat of the target's 3 aux hidden states [3 * H_t] -> H   (combine_hidden_states)
- layer 0 [LlamaDecoderLayer, layer_idx == 0]:
    embeds  = input_layernorm(embed(token))
    residual = hidden;  hidden = hidden_norm(hidden)   (norm_before_residual: swapped)
    h = self_attn(concat[embeds, hidden])              (q/k/v take 2H inputs)
    residual = h + residual;  h = post_attention_layernorm(residual)
    h = mlp(h)
- final: prenorm = h + residual;  out = norm(prenorm)             (LlamaModel.forward)
- logits over the draft vocab from `out`; draft id d -> target id d + d2t[d]
- the next chained step's hidden input is `prenorm` (aux_output, norm_output=False)

The head carries no embedding of its own (none in the checkpoints we use):
like vLLM, it shares the target's embed_tokens, loaded from the target
checkpoint.

Attention runs over explicit per-sequence KV tensors so the caller controls
rollback (truncate) and re-prefill; batched by padding with a mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


@dataclass
class StepInput:
    """One sequence's slice of a batched forward."""
    token_ids: torch.Tensor      # [q] target-vocab ids
    hidden: torch.Tensor         # [q, H] (already fc'd, or the previous step's prenorm)
    positions: torch.Tensor      # [q]
    past_k: torch.Tensor         # [n_kv, p, d]
    past_v: torch.Tensor


class Eagle3Head(nn.Module):
    def __init__(self, config, target_hidden_size: int, num_aux: int = 3) -> None:
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        c = config
        self.config = c
        H = c.hidden_size
        self.nh = c.num_attention_heads
        self.nkv = c.num_key_value_heads
        self.hd = getattr(c, "head_dim", None) or H // self.nh
        eps = c.rms_norm_eps
        self.embed_tokens = nn.Embedding(c.vocab_size, H)
        self.fc = nn.Linear(num_aux * target_hidden_size, H, bias=False)
        self.input_layernorm = RMSNorm(H, eps)
        self.hidden_norm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.norm = RMSNorm(H, eps)
        bias = getattr(c, "attention_bias", False)
        self.q_proj = nn.Linear(2 * H, self.nh * self.hd, bias=bias)
        self.k_proj = nn.Linear(2 * H, self.nkv * self.hd, bias=bias)
        self.v_proj = nn.Linear(2 * H, self.nkv * self.hd, bias=bias)
        self.o_proj = nn.Linear(self.nh * self.hd, H, bias=False)
        self.gate_proj = nn.Linear(H, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(H, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, H, bias=False)
        draft_vocab = getattr(c, "draft_vocab_size", None) or c.vocab_size
        self.lm_head = nn.Linear(H, draft_vocab, bias=False)
        self.register_buffer("d2t", torch.zeros(draft_vocab, dtype=torch.long))
        self.norm_before_residual = bool(getattr(c, "norm_before_residual", False))
        self.rotary = LlamaRotaryEmbedding(config=c)

    # -- weights -----------------------------------------------------------
    _NAME_MAP = {
        "midlayer.self_attn.q_proj.weight": "q_proj.weight",
        "midlayer.self_attn.k_proj.weight": "k_proj.weight",
        "midlayer.self_attn.v_proj.weight": "v_proj.weight",
        "midlayer.self_attn.o_proj.weight": "o_proj.weight",
        "midlayer.mlp.gate_proj.weight": "gate_proj.weight",
        "midlayer.mlp.up_proj.weight": "up_proj.weight",
        "midlayer.mlp.down_proj.weight": "down_proj.weight",
        "midlayer.hidden_norm.weight": "hidden_norm.weight",
        "midlayer.input_layernorm.weight": "input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "norm.weight": "norm.weight",
        "fc.weight": "fc.weight",
        "lm_head.weight": "lm_head.weight",
        "d2t": "d2t",
        "embed_tokens.weight": "embed_tokens.weight",
    }

    def load_head_state(self, sd: dict[str, torch.Tensor]) -> None:
        own = {}
        for k, v in sd.items():
            if k == "t2d":
                continue
            if k not in self._NAME_MAP:
                raise KeyError(f"unexpected EAGLE3 weight {k!r}")
            own[self._NAME_MAP[k]] = v
        if "d2t" not in own:  # draft vocab == target vocab
            own["d2t"] = torch.zeros_like(self.d2t)
        missing = set(dict(self.named_parameters())) | {"d2t"}
        missing -= set(own)
        missing.discard("embed_tokens.weight")  # shared from the target, set separately
        if missing:
            raise KeyError(f"EAGLE3 checkpoint lacks {sorted(missing)}")
        self.load_state_dict(own, strict=False)

    # -- math --------------------------------------------------------------
    def combine(self, aux: torch.Tensor) -> torch.Tensor:
        """[n, 3 * H_t] target aux hidden states -> [n, H]."""
        return self.fc(aux.to(self.fc.weight.dtype))

    def to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]

    @torch.no_grad()
    def forward_batch(self, items: list[StepInput]):
        """-> (out [B, Tq, H], prenorm [B, Tq, H], new_k list, new_v list).

        Row i's valid outputs are the first len(items[i].token_ids) positions.
        new_k[i]/new_v[i] are the K/V of those positions ([n_kv, q, d]), for
        the caller to append to that sequence's cache.
        """
        dev, dt = self.fc.weight.device, self.fc.weight.dtype
        B = len(items)
        qs = [len(it.token_ids) for it in items]
        ps = [it.past_k.shape[1] for it in items]
        Tq = max(qs)
        Tk = max(p + q for p, q in zip(ps, qs))
        H = self.config.hidden_size

        tok = torch.zeros(B, Tq, dtype=torch.long, device=dev)
        hid = torch.zeros(B, Tq, H, dtype=dt, device=dev)
        pos = torch.zeros(B, Tq, dtype=torch.long, device=dev)
        for i, it in enumerate(items):
            tok[i, :qs[i]] = it.token_ids
            hid[i, :qs[i]] = it.hidden.to(dt)
            pos[i, :qs[i]] = it.positions

        embeds = self.input_layernorm(self.embed_tokens(tok))
        if self.norm_before_residual:
            hid = self.hidden_norm(hid)
            residual = hid
        else:
            residual = hid
            hid = self.hidden_norm(hid)
        x = torch.cat([embeds, hid], dim=-1)

        q = self.q_proj(x).view(B, Tq, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, Tq, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, Tq, self.nkv, self.hd).transpose(1, 2)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = self.rotary(v, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        K = torch.zeros(B, self.nkv, Tk, self.hd, dtype=dt, device=dev)
        V = torch.zeros_like(K)
        mask = torch.zeros(B, 1, Tq, Tk, dtype=torch.bool, device=dev)
        new_k, new_v = [], []
        for i, it in enumerate(items):
            p, n = ps[i], qs[i]
            if p:
                K[i, :, :p] = it.past_k.to(dt)
                V[i, :, :p] = it.past_v.to(dt)
            K[i, :, p:p + n] = k[i, :, :n]
            V[i, :, p:p + n] = v[i, :, :n]
            new_k.append(k[i, :, :n])
            new_v.append(v[i, :, :n])
            # query j (absolute p + j) sees keys 0 .. p + j
            causal = torch.arange(Tk, device=dev)[None, :] <= (p + torch.arange(Tq, device=dev))[:, None]
            causal[n:] = False
            causal[n:, 0] = True  # padded query rows: keep softmax finite, output ignored
            mask[i, 0] = causal
        rep = self.nh // self.nkv
        attn = F.scaled_dot_product_attention(
            q, K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1), attn_mask=mask)
        h = self.o_proj(attn.transpose(1, 2).reshape(B, Tq, self.nh * self.hd))

        residual = h + residual
        h = self.post_attention_layernorm(residual)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        prenorm = h + residual
        out = self.norm(prenorm)
        return out, prenorm, new_k, new_v

    def logits(self, out: torch.Tensor) -> torch.Tensor:
        return self.lm_head(out)


def load_eagle3_head(head_path: str, target_path: str, device: str = "cuda",
                     dtype: torch.dtype = torch.bfloat16) -> Eagle3Head:
    """head_path / target_path: local snapshot dirs (or hub ids, resolved
    through the HF cache)."""
    from transformers import AutoConfig, LlamaConfig

    head_dir = _resolve(head_path)
    target_dir = _resolve(target_path)
    cfg = LlamaConfig.from_pretrained(head_dir)
    tcfg = AutoConfig.from_pretrained(target_dir)
    head = Eagle3Head(cfg, target_hidden_size=getattr(cfg, "target_hidden_size", None) or tcfg.hidden_size)

    sd = _load_state(head_dir)
    head.load_head_state(sd)
    if "embed_tokens.weight" not in sd:
        head.embed_tokens.weight.data.copy_(_load_tensor(target_dir, "model.embed_tokens.weight"))
    return head.to(device=device, dtype=dtype).eval()


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path))


def _load_state(d: Path) -> dict[str, torch.Tensor]:
    st = sorted(d.glob("*.safetensors"))
    if st:
        from safetensors.torch import load_file
        out = {}
        for f in st:
            out.update(load_file(str(f)))
        return out
    return torch.load(d / "pytorch_model.bin", map_location="cpu", weights_only=True)


def _load_tensor(d: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open
    idx = d / "model.safetensors.index.json"
    files = ([json.loads(idx.read_text())["weight_map"][name]] if idx.exists()
             else [f.name for f in d.glob("*.safetensors")])
    for f in files:
        with safe_open(str(d / f), framework="pt") as fh:
            if name in fh.keys():
                return fh.get_tensor(name)
    raise KeyError(f"{name} not found in {d}")
