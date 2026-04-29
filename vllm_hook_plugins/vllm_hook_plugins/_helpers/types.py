"""Public schema for activation steering payloads.

A SteeringVector carries one or more residual-stream additions tagged with
the layer indices they apply to. The tensor field round-trips through
serialize_tensor/deserialize_tensor so the model can be passed through
JSON-only transports.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from ._serialize import deserialize_tensor, serialize_tensor


class SteeringVector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # (n_layers, hidden_dim) or (n_layers, n_pos, hidden_dim).
    activations: torch.Tensor
    # len(layer_indices) must equal activations.shape[0].
    layer_indices: list[int]
    # Multiplier on the steering contribution.
    scale: float = 1.0
    # If true, scale the steering vector to match the residual's L2 norm
    # before adding (per-token). If false, add raw vec * scale.
    norm_match: bool = False
    # Absolute token positions to apply at; None means every token in the
    # current request slice.
    position_indices: list[int] | None = None

    @field_serializer("activations")
    def _serialize(self, v, _info):
        return serialize_tensor(v)

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize(cls, v: Any) -> torch.Tensor:
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(f"Cannot interpret as a tensor: {type(v).__name__}")
