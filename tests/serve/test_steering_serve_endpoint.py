"""End-to-end test through the OpenAI-compatible HTTP server."""

from __future__ import annotations

import json

import pytest

vllm = pytest.importorskip("vllm")
httpx = pytest.importorskip("httpx")
import torch

from vllm_hook_plugins._helpers.types import SteeringVector


def _hidden_size_for(model: str) -> int:
    sizes = {
        "facebook/opt-125m": 768,
        "gpt2": 768,
        "Qwen/Qwen2-1.5B-Instruct": 1536,
    }
    return sizes.get(model, 4096)


@pytest.mark.gpu
@pytest.mark.serve
def test_steering_via_openai_endpoint(serve_url, small_model):
    torch.manual_seed(0)
    sv = SteeringVector(
        activations=(torch.randn(1, _hidden_size_for(small_model)) * 5.0).to(
            torch.bfloat16
        ),
        layer_indices=[3],
        scale=1.0,
    )

    body_clean = {
        "model": small_model,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "max_tokens": 30,
    }
    body_steered = dict(body_clean)
    body_steered["extra_body"] = {
        "vllm_xargs": {"apply_steering_vectors": json.dumps([sv.model_dump()])}
    }

    r = httpx.post(
        f"{serve_url}/v1/chat/completions", json=body_steered, timeout=120.0
    )
    r.raise_for_status()
    steered = r.json()["choices"][0]["message"]["content"]

    r = httpx.post(
        f"{serve_url}/v1/chat/completions", json=body_clean, timeout=120.0
    )
    r.raise_for_status()
    clean = r.json()["choices"][0]["message"]["content"]

    assert steered != clean
