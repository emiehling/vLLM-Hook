"""Worker-side mixin for per-request residual steering.

vLLM injects this class into each Worker via `Worker.__bases__ += (cls,)`
when `worker_extension_cls` is set, so the methods defined here become
callable through `engine.collective_rpc(...)`. Steering data is keyed by
request id and looked up from inside a forward hook on every transformer
block.
"""

from __future__ import annotations

import logging
import pickle

import torch

logger = logging.getLogger("vllm_hook_plugins.steering")


def _get_layers(model):
    m = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(
        m.model.decoder, "layers"
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise AttributeError(
        f"Cannot resolve transformer layers for model {type(m).__name__}"
    )


def _apply_steering_to_residual(
    steering_vectors,
    layer_idx: int,
    residual: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
):
    """Apply each matching SteeringVector in-place on residual[start:end]."""
    for sv in steering_vectors:
        if layer_idx not in sv.layer_indices:
            continue
        slot = sv.layer_indices.index(layer_idx)
        vec = sv.activations[slot].to(residual.device, dtype=residual.dtype)

        if vec.dim() != 1:
            # Per-position steering tensor (n_pos, hidden) is not supported in
            # the first cut; skip with a warning so a misconfigured request
            # doesn't silently drop on the floor.
            logger.warning(
                "skipping multi-position steering at layer %d (vec.dim=%d)",
                layer_idx,
                vec.dim(),
            )
            continue

        if sv.position_indices is None:
            target = residual[start:end]
            if sv.norm_match:
                eps = 1e-6
                r_norm = target.float().norm(dim=-1, keepdim=True)
                v_norm = vec.float().norm() + eps
                contrib = (vec * (r_norm / v_norm)).to(residual.dtype) * sv.scale
            else:
                contrib = vec * sv.scale
            residual[start:end] = target + contrib
            continue

        local_idx = [
            p - abs_start
            for p in sv.position_indices
            if 0 <= (p - abs_start) < (end - start)
        ]
        if not local_idx:
            continue
        idx_tensor = torch.tensor(local_idx, device=residual.device, dtype=torch.long)
        view = residual[start:end]
        target = view.index_select(0, idx_tensor)
        if sv.norm_match:
            eps = 1e-6
            r_norm = target.float().norm(dim=-1, keepdim=True)
            v_norm = vec.float().norm() + eps
            contrib = (vec * (r_norm / v_norm)).to(residual.dtype) * sv.scale
        else:
            contrib = vec * sv.scale
        view.index_copy_(0, idx_tensor, target + contrib)


def _make_hook(extension, layer_idx: int):
    def hook(_module, _args, output):
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.exception("steering hook failed at layer %d", layer_idx)
            return None

    return hook


def _hook_inner(extension, layer_idx: int, output):
    # Zero-cost short-circuit when no request in the current batch is steering.
    if not extension._steering_data:
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None

    runner = getattr(extension, "model_runner", None)
    if runner is None:
        return None
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return None
    num_reqs = getattr(input_batch, "num_reqs", 0)
    if num_reqs == 0:
        return None

    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)
    if metadata is None:
        return None

    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if query_start_loc is None and isinstance(metadata, dict):
        for entry in metadata.values():
            qsl = getattr(entry, "query_start_loc", None)
            if qsl is not None:
                query_start_loc = qsl
                seq_lens = getattr(entry, "seq_lens", None)
                break
    if query_start_loc is None:
        return None

    req_ids = list(input_batch.req_ids[:num_reqs])

    per_req: list[list] = []
    needs_steering = False
    requests_dict = getattr(runner, "requests", {}) or {}
    for req_id in req_ids:
        cfgs = extension._steering_data.get(req_id)
        if cfgs is None:
            req_state = requests_dict.get(req_id)
            sp = getattr(req_state, "sampling_params", None) if req_state else None
            extra = getattr(sp, "extra_args", None) if sp else None
            sid = (extra or {}).get("_steering_id")
            if sid is not None:
                cfgs = extension._steering_data.get(sid)
        per_req.append(cfgs or [])
        if cfgs:
            needs_steering = True

    if not needs_steering:
        return None

    is_tuple = isinstance(output, tuple)
    if is_tuple:
        hidden_states, residual = output[0], output[1]
        if not isinstance(residual, torch.Tensor):
            return None
        residual = residual.clone()
    else:
        if not isinstance(output, torch.Tensor):
            return None
        hidden_states = None
        residual = output.clone()

    for i in range(num_reqs):
        if not per_req[i]:
            continue
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        n_query = end - start
        sl = seq_lens[i] if seq_lens is not None else None
        if isinstance(sl, torch.Tensor):
            sl_val = int(sl.item())
        elif sl is not None:
            sl_val = int(sl)
        else:
            sl_val = end
        abs_start = sl_val - n_query
        _apply_steering_to_residual(
            per_req[i], layer_idx, residual, start, end, abs_start
        )

    return (hidden_states, residual) if is_tuple else residual


class SteeringExtension:
    """Mixin appended to vLLM Worker via worker_extension_cls.

    Worker construction is owned by vLLM; this class deliberately does not
    define __init__. Per-instance state is created lazily on the first call
    to install_hooks().
    """

    def install_hooks(self):
        if getattr(self, "_steering_hooks_installed", False):
            return
        self._steering_data: dict[str, list] = {}

        from vllm.model_executor.models.utils import PPMissingLayer

        layers = _get_layers(self.model_runner.model)
        installed = 0
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            layer.register_forward_hook(_make_hook(self, layer_idx))
            installed += 1

        self._steering_hooks_installed = True
        logger.info("installed steering hooks on %d layers", installed)

    def set_steering_data(self, request_id: str, pickled: bytes):
        if not hasattr(self, "_steering_data"):
            self._steering_data = {}
        self._steering_data[request_id] = pickle.loads(pickled)

    def clear_steering_data(self, request_id: str):
        if hasattr(self, "_steering_data"):
            self._steering_data.pop(request_id, None)
