"""Engine-side patches for per-request activation steering.

register() runs once at engine startup. It (1) injects the steering
WorkerExtension as the default worker_extension_cls when none is set, and
(2) wraps AsyncLLM.generate and LLM.generate so steering payloads carried
in SamplingParams.extra_args are extracted, RPC'd to the workers, and
torn down again after the request completes.

The wire format key is "apply_steering_vectors" inside extra_args. At the
HTTP layer, vLLM's OpenAI server forwards `extra_body.vllm_xargs` into
SamplingParams.extra_args, so a client posts:

    extra_body: {"vllm_xargs": {"apply_steering_vectors": "<json string>"}}
"""

from __future__ import annotations

import json
import logging
import pickle
import uuid

logger = logging.getLogger("vllm_hook_plugins.steering")

_PATCHED = False
_EAGER_WARNING_EMITTED = False

_EXTENSION_PATH = "vllm_hook_plugins._steering_extension.SteeringExtension"
_REQ_KEY = "apply_steering_vectors"
_SID_KEY = "_steering_id"
_SKIP_PREFIX_KEY = "skip_reading_prefix_cache"


def register():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError:
        # vLLM not present (e.g. only the wire-format helpers are being used).
        # Nothing to patch.
        return

    _patch_engine_args(EngineArgs)
    AsyncLLM.generate = _wrap_async_generate(AsyncLLM.generate)
    LLM.generate = _wrap_llm_generate(LLM.generate)


def _patch_engine_args(EngineArgs):
    orig = EngineArgs.create_engine_config

    def patched(self, *args, **kwargs):
        if not getattr(self, "worker_extension_cls", None):
            self.worker_extension_cls = _EXTENSION_PATH
        return orig(self, *args, **kwargs)

    EngineArgs.create_engine_config = patched


def _wrap_async_generate(orig):
    async def patched(self, prompt, sampling_params, request_id, *args, **kwargs):
        steering, sampling_params = _extract_steering(sampling_params)
        if steering is not None:
            _warn_if_not_eager(self)
            await _ensure_hooks_installed_async(self)
            await self.collective_rpc(
                "set_steering_data", args=(request_id, pickle.dumps(steering))
            )
            _set_skip_prefix(sampling_params)

        try:
            async for output in orig(
                self, prompt, sampling_params, request_id, *args, **kwargs
            ):
                yield output
        finally:
            if steering is not None:
                try:
                    await self.collective_rpc(
                        "clear_steering_data", args=(request_id,)
                    )
                except Exception:
                    logger.exception("failed to clear steering data for %s", request_id)

    return patched


def _wrap_llm_generate(orig):
    """LLM.generate allocates request_ids internally, so steering payloads
    are tagged with a synthetic id stuffed into extra_args; the worker
    resolves it back from the request state inside the hook."""

    def patched(self, *args, **kwargs):
        prompts = _peek_arg(args, kwargs, 0, "prompts")
        sampling_params = _peek_arg(args, kwargs, 1, "sampling_params")

        is_list = isinstance(sampling_params, list)
        if is_list:
            params_list = sampling_params
        else:
            n = len(prompts) if isinstance(prompts, list) else 1
            params_list = [sampling_params] * n

        cleaned: list = []
        steering_ids: list[tuple[str, list]] = []
        for sp in params_list:
            steering, sp_clean = _extract_steering(sp)
            if steering is not None:
                sid = f"_steering_{uuid.uuid4().hex}"
                steering_ids.append((sid, steering))
                _stash_steering_id(sp_clean, sid)
                _set_skip_prefix(sp_clean)
            cleaned.append(sp_clean)

        if not steering_ids:
            # Fast path: nothing to clean, hand off to vLLM untouched.
            return orig(self, *args, **kwargs)

        new_sp = cleaned if is_list else cleaned[0]
        args, kwargs = _replace_arg(args, kwargs, 1, "sampling_params", new_sp)

        _warn_if_not_eager(self)
        _ensure_hooks_installed_sync(self)
        for sid, steering in steering_ids:
            self.collective_rpc(
                "set_steering_data", args=(sid, pickle.dumps(steering))
            )

        try:
            return orig(self, *args, **kwargs)
        finally:
            for sid, _ in steering_ids:
                try:
                    self.collective_rpc("clear_steering_data", args=(sid,))
                except Exception:
                    logger.exception("failed to clear steering data for %s", sid)

    return patched


def _peek_arg(args, kwargs, position: int, name: str):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def _replace_arg(args, kwargs, position: int, name: str, value):
    if name in kwargs:
        kwargs = dict(kwargs)
        kwargs[name] = value
        return args, kwargs
    if len(args) > position:
        args = args[:position] + (value,) + args[position + 1 :]
        return args, kwargs
    kwargs = dict(kwargs)
    kwargs[name] = value
    return args, kwargs


def _extract_steering(sampling_params):
    """Pop apply_steering_vectors from extra_args.

    Returns (parsed_list_or_None, sampling_params).
    Tensors aren't valid msgspec values, so this MUST happen before the
    sampling_params crosses any msgspec boundary.
    """
    from ._helpers.types import SteeringVector

    if sampling_params is None:
        return None, sampling_params
    extra = getattr(sampling_params, "extra_args", None)
    if not extra:
        return None, sampling_params
    raw = extra.pop(_REQ_KEY, None)
    if raw is None:
        return None, sampling_params

    if isinstance(raw, str):
        raw = json.loads(raw)

    parsed = [SteeringVector.model_validate(item) for item in raw]
    return parsed, sampling_params


def _stash_steering_id(sampling_params, sid: str):
    extra = getattr(sampling_params, "extra_args", None)
    if extra is None:
        sampling_params.extra_args = {_SID_KEY: sid}
    else:
        extra[_SID_KEY] = sid


def _set_skip_prefix(sampling_params):
    extra = getattr(sampling_params, "extra_args", None)
    if extra is None:
        sampling_params.extra_args = {_SKIP_PREFIX_KEY: True}
    else:
        extra[_SKIP_PREFIX_KEY] = True


async def _ensure_hooks_installed_async(engine):
    if getattr(engine, "_steering_hooks_installed", False):
        return
    await engine.collective_rpc("install_hooks")
    engine._steering_hooks_installed = True


def _ensure_hooks_installed_sync(engine):
    if getattr(engine, "_steering_hooks_installed", False):
        return
    engine.collective_rpc("install_hooks")
    engine._steering_hooks_installed = True


def _warn_if_not_eager(engine):
    global _EAGER_WARNING_EMITTED
    if _EAGER_WARNING_EMITTED:
        return
    cfg = getattr(engine, "vllm_config", None)
    enforce_eager = None
    for chain in (
        ("model_config", "enforce_eager"),
        ("compilation_config", "enforce_eager"),
    ):
        owner = cfg
        for attr in chain:
            owner = getattr(owner, attr, None) if owner else None
        if owner is not None:
            enforce_eager = owner
            break
    if enforce_eager is False:
        logger.warning(
            "activation steering requested but engine has enforce_eager=False; "
            "forward hooks do not fire inside captured CUDA graphs and steering "
            "may be silently skipped. Pass enforce_eager=True to the engine."
        )
        _EAGER_WARNING_EMITTED = True
