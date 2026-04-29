"""Wire-format round-trip tests. CPU-only; no vLLM import required."""

from __future__ import annotations

import json

import pytest
import torch

from vllm_hook_plugins._helpers._serialize import (
    deserialize_tensor,
    serialize_tensor,
)
from vllm_hook_plugins._helpers.types import SteeringVector


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32, torch.float64],
)
def test_tensor_round_trip_bit_exact(dtype):
    torch.manual_seed(0)
    if dtype is torch.bfloat16:
        # bf16 has limited precision; round to representable values then view
        # so the comparison is bit-exact rather than approximate.
        original = torch.randn(4, 16).to(dtype)
    else:
        original = torch.randn(4, 16, dtype=dtype)

    payload = serialize_tensor(original)
    restored = deserialize_tensor(payload)

    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert torch.equal(restored, original)


def test_tensor_round_trip_through_json():
    original = (torch.randn(2, 8) * 5).to(torch.bfloat16)
    payload = serialize_tensor(original)
    blob = json.dumps(payload)
    restored = deserialize_tensor(json.loads(blob))
    assert torch.equal(restored, original)


def test_steering_vector_model_dump_round_trip():
    sv = SteeringVector(
        activations=(torch.randn(2, 16) * 2).to(torch.bfloat16),
        layer_indices=[3, 7],
        scale=0.5,
        norm_match=True,
        position_indices=[0, 1, 2],
    )

    blob = json.dumps(sv.model_dump())
    restored = SteeringVector.model_validate(json.loads(blob))

    assert restored.layer_indices == sv.layer_indices
    assert restored.scale == sv.scale
    assert restored.norm_match == sv.norm_match
    assert restored.position_indices == sv.position_indices
    assert restored.activations.dtype == torch.bfloat16
    assert torch.equal(restored.activations, sv.activations)


def test_steering_vector_accepts_tensor_passthrough():
    """In-process callers can hand SteeringVector a torch.Tensor directly,
    bypassing the serialize/deserialize round trip."""
    t = torch.randn(1, 8, dtype=torch.float32)
    sv = SteeringVector(activations=t, layer_indices=[0])
    assert sv.activations is t


def test_steering_vector_rejects_garbage():
    with pytest.raises(Exception):
        SteeringVector(activations="not a tensor", layer_indices=[0])


def test_zero_size_tensor():
    original = torch.zeros(0, 16, dtype=torch.bfloat16)
    payload = serialize_tensor(original)
    restored = deserialize_tensor(payload)
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
