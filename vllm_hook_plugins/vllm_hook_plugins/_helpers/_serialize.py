"""Tensor <-> JSON-safe dict codec.

The wire format encodes a tensor as a dict of JSON primitives so it can
travel through msgspec-validated fields like SamplingParams.extra_args.
bfloat16 has no native numpy dtype, so it is viewed as int16 for transport
and reinterpreted on the receiving side.
"""

from __future__ import annotations

import base64

import numpy as np
import torch
import zstandard

_ZSTD_COMPRESSOR = zstandard.ZstdCompressor()
_ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()

_DTYPE_REVERSE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
}


def serialize_tensor(t: torch.Tensor) -> dict:
    t = t.detach().cpu().contiguous()
    original_dtype = str(t.dtype)
    if t.dtype is torch.bfloat16:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()
    raw = arr.tobytes()
    compressed = _ZSTD_COMPRESSOR.compress(raw)
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }


def deserialize_tensor(d: dict) -> torch.Tensor:
    raw = _ZSTD_DECOMPRESSOR.decompress(base64.b64decode(d["data"]))
    np_dtype = np.dtype(d["dtype"])
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(d["shape"])
    # frombuffer returns a read-only view; copy so the resulting tensor is writable.
    t = torch.from_numpy(arr.copy())
    target = _DTYPE_REVERSE.get(d["original_dtype"])
    if target is None:
        return t
    if target is torch.bfloat16:
        return t.view(torch.bfloat16)
    return t.to(target)
