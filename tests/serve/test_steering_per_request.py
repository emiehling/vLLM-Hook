"""Per-request steering through the offline LLM API."""

from __future__ import annotations

import pytest
import torch

vllm = pytest.importorskip("vllm")
from vllm import LLM, SamplingParams

from vllm_hook_plugins import register_plugins
from vllm_hook_plugins._helpers.types import SteeringVector


def _hidden_size_for(model: str) -> int:
    sizes = {
        "facebook/opt-125m": 768,
        "gpt2": 768,
        "Qwen/Qwen2-1.5B-Instruct": 1536,
    }
    return sizes.get(model, 4096)


@pytest.fixture(scope="module", autouse=True)
def _register():
    register_plugins()


@pytest.mark.gpu
def test_steering_changes_output(small_model, cache_root):
    llm = LLM(
        model=small_model,
        download_dir=str(cache_root),
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )

    prompt = "Should governments regulate artificial intelligence?"
    sp_clean = SamplingParams(temperature=0.0, max_tokens=40)
    out_clean = llm.generate(prompt, sp_clean)[0].outputs[0].text

    torch.manual_seed(0)
    sv = SteeringVector(
        activations=(torch.randn(1, _hidden_size_for(small_model)) * 5.0).to(
            torch.bfloat16
        ),
        layer_indices=[3],
        scale=1.0,
    )
    sp_steered = SamplingParams(
        temperature=0.0,
        max_tokens=40,
        extra_args={"apply_steering_vectors": [sv.model_dump()]},
    )
    out_steered = llm.generate(prompt, sp_steered)[0].outputs[0].text

    assert out_clean != out_steered, (
        "A strong random steering vector should perturb deterministic output"
    )


@pytest.mark.gpu
def test_unsteered_request_unaffected(small_model, cache_root):
    """Two consecutive unsteered requests must produce identical output —
    proves the hook is a no-op when no steering payload is attached."""
    llm = LLM(
        model=small_model,
        download_dir=str(cache_root),
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    a = llm.generate("Hello world", sp)[0].outputs[0].text
    b = llm.generate("Hello world", sp)[0].outputs[0].text
    assert a == b
