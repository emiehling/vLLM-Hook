"""Concurrent steered + unsteered requests must not bleed into each other.

This is the truth-test of the per-request opt-in design: an unsteered
request batched alongside a steered one must produce byte-identical
output to a baseline run with no steering at all.
"""

from __future__ import annotations

import asyncio
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
def test_concurrent_steered_and_unsteered(serve_url, small_model):
    torch.manual_seed(0)
    sv = SteeringVector(
        activations=(torch.randn(1, _hidden_size_for(small_model)) * 5.0).to(
            torch.bfloat16
        ),
        layer_indices=[3],
        scale=1.0,
    )
    base = {
        "model": small_model,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "max_tokens": 30,
    }
    clean_body = dict(base)
    steered_body = dict(
        base,
        extra_body={
            "vllm_xargs": {"apply_steering_vectors": json.dumps([sv.model_dump()])}
        },
    )

    async def fire(client: httpx.AsyncClient, body: dict) -> str:
        r = await client.post(f"{serve_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def main():
        async with httpx.AsyncClient(timeout=120.0) as c:
            a = await fire(c, clean_body)
            b = await fire(c, clean_body)
            assert a == b, "baseline must be deterministic with temperature=0"

            results = await asyncio.gather(
                fire(c, steered_body), fire(c, clean_body),
                fire(c, steered_body), fire(c, clean_body),
                fire(c, steered_body), fire(c, clean_body),
                fire(c, steered_body), fire(c, clean_body),
            )
            steered = results[0::2]
            unsteered = results[1::2]
            assert all(u == a for u in unsteered), (
                "unsteered output must equal baseline even when batched with steered "
                "requests"
            )
            assert all(s != a for s in steered)

    asyncio.run(main())
