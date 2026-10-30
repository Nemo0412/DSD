"""Pack EAGLE tree tensors into SpeculateRequest / SpeculateReply fields."""
from __future__ import annotations

import torch

from dspec_real.proto.messages import SpeculateReply

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def pack_hidden(hidden: torch.Tensor) -> tuple[bytes, list[int], int, str]:
    t = hidden.detach().cpu().contiguous()
    raw = t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
    return raw, list(t.shape), t.shape[-1], dtype_name(t.dtype)


def unpack_hidden(raw: bytes | None, shape: list[int], dim: int, dtype: str) -> torch.Tensor:
    if not raw:
        return torch.empty(0, dim)
    dt = _DTYPES[dtype or "bfloat16"]
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    if dt == torch.bfloat16:
        values = flat.view(dt)
    else:
        values = flat.view(dt)
    if shape:
        return values.reshape(shape)
    n = values.numel() // dim
    return values.reshape(1, n, dim)


def pack_tree(draft_tokens: torch.Tensor, retrieve: torch.Tensor,
              tree_mask: torch.Tensor, positions: torch.Tensor) -> dict:
    mask = tree_mask.detach().cpu().contiguous().float()
    retrieve = retrieve.detach().cpu().contiguous().long()
    positions = positions.detach().cpu().contiguous().long().view(-1)
    tokens = draft_tokens.detach().cpu().contiguous().long().view(-1)
    return {
        "draft_tokens": [int(x) for x in tokens.tolist()],
        "tree_mask": mask.numpy().tobytes(),
        "tree_mask_shape": list(mask.shape),
        "tree_mask_dtype": "float32",
        "tree_position_ids": [int(x) for x in positions.tolist()],
        "retrieve_indices": [int(x) for x in retrieve.view(-1).tolist()],
        "retrieve_shape": list(retrieve.shape),
    }


def unpack_tree(reply: SpeculateReply, device: torch.device | str):
    if not reply.tree_mask or not reply.tree_mask_shape:
        raise ValueError(f"drafter {reply.seq_id} returned no tree")
    mask = torch.frombuffer(bytearray(reply.tree_mask), dtype=torch.float32).reshape(reply.tree_mask_shape)
    retrieve = torch.tensor(reply.retrieve_indices, dtype=torch.long).reshape(reply.retrieve_shape)
    positions = torch.tensor(reply.tree_position_ids, dtype=torch.long)
    tokens = torch.tensor(reply.draft_tokens, dtype=torch.long).view(1, -1)
    dev = torch.device(device)
    return tokens.to(dev), retrieve.to(dev), mask.to(dev), positions.to(dev)
