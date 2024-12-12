from typing import Tuple
from transformers import AutoModelForCausalLM

import torch
import torch.nn as nn
import torch.nn.functional as F

# ref: https://medium.com/@govindarajpriyanthan/build-an-8-bit-custom-quantizer-from-scratch-a-comprehensive-guide-69d34486f89a
class Qint_Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        goal_dtype: torch.dtype = torch.int8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        self.in_dtype = dtype
        assert goal_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        self.out_dtype = goal_dtype

        self.NEW_MIN, self.NEW_MAX = (
            torch.iinfo(goal_dtype).min,
            torch.iinfo(goal_dtype).max,
        )

        self.register_buffer(
            "weights",
            torch.randint(
                self.NEW_MIN,
                self.NEW_MAX,
                (out_features, in_features),
                dtype=goal_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "scales", torch.randn((out_features), dtype=dtype, device=self.device)
        )
        if bias:
            self.register_buffer(
                "bias", torch.randn((1, out_features), dtype=dtype, device=self.device)
            )
        else:
            self.bias = None

    def quantize(self, weights: torch.Tensor):
        weights_in = weights.clone().to(self.in_dtype)
        scales = weights_in.abs().max(dim=1).values / self.NEW_MAX
        scales = scales.to(weights.dtype)
        self.scales = scales.to(self.device)

        quantized_weights = torch.round(weights / scales[:, None]).to(self.out_dtype)
        self.weights = quantized_weights.to(self.device)

    def forward(self, x: torch.Tensor):
        converted_weights = self.weights.to(x.dtype)
        output = F.linear(x, converted_weights) * self.scales

        if self.bias is not None:
            output += self.bias

        return output


class Qint_Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        dtype: torch.dtype = torch.float32,
        goal_dtype: torch.dtype = torch.int8,
        device: torch.device = torch.device("cpu"),
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()

        self.device = device
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.in_dtype = dtype
        assert goal_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        self.out_dtype = goal_dtype

        self.NEW_MIN, self.NEW_MAX = (
            torch.iinfo(goal_dtype).min,
            torch.iinfo(goal_dtype).max,
        )

        self.register_buffer(
            "weights",
            torch.randint(
                self.NEW_MIN,
                self.NEW_MAX,
                (num_embeddings, embedding_dim),
                dtype=goal_dtype,
                device=device,
            ),
        )

        self.register_buffer(
            "scales", torch.randn((embedding_dim), dtype=dtype, device=self.device)
        )

        self.padding_idx = padding_idx

    def quantize(self, weights: torch.Tensor):
        weights_in = weights.clone().to(self.in_dtype)
        scales = weights_in.abs().max(dim=1).values / self.NEW_MAX
        scales = scales.to(weights.dtype)
        self.scales = scales.to(self.device)

        quantized_weights = torch.round(weights / scales[:, None]).to(self.out_dtype)
        self.weights = quantized_weights.to(self.device)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        converted_weights = self.weights.to(x.dtype) * self.scales.unsqueeze(1)
        output = F.embedding(
            x,
            converted_weights,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        return output


class Qint_LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Tuple,
        dtype: torch.dtype,
        goal_dtype: torch.dtype,
        device: torch.device,
        bias: bool = True,
    ):
        super().__init__()

        self.device = device

        self.normalized_shape = normalized_shape

        self.in_dtype = dtype
        assert goal_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        self.out_dtype = goal_dtype

        self.NEW_MIN, self.NEW_MAX = (
            torch.iinfo(goal_dtype).min,
            torch.iinfo(goal_dtype).max,
        )

        self.register_buffer(
            "weights",
            torch.randint(
                self.NEW_MIN,
                self.NEW_MAX,
                normalized_shape,
                dtype=goal_dtype,
                device=device,
            ),
        )

        self.register_buffer(
            "scales", torch.randn(normalized_shape, dtype=dtype, device=self.device)
        )

        if bias:
            self.register_buffer(
                "bias", torch.randn(normalized_shape, dtype=dtype, device=self.device)
            )
        else:
            self.bias = None

    def quantize(self, weights: torch.Tensor):
        weights_in = weights.clone().to(self.in_dtype)
        scales = weights_in.abs().max(dim=-1).values / self.NEW_MAX
        scales = scales.to(weights.dtype)
        self.scales = scales.to(self.device)

        quantized_weights = torch.round(weights / scales).to(self.out_dtype)
        self.weights = quantized_weights.to(self.device)

    def forward(self, x: torch.Tensor):
        converted_weights = self.weights.to(x.dtype) * self.scales
        output = F.layer_norm(x, self.normalized_shape, weight=converted_weights)

        if self.bias is not None:
            output += self.bias

        return output


def quantize_linear(
    module: nn.Module | AutoModelForCausalLM,
    name: str,
    child: nn.Linear,
    goal_dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
):
    old_bias, old_weights = child.bias, child.weight
    has_bias = old_bias is not None
    quantized_layer = Qint_Linear(
        child.in_features,
        child.out_features,
        has_bias,
        child.weight.dtype,
        goal_dtype,
        device,
    )

    setattr(module, name, quantized_layer)

    getattr(module, name).quantize(old_weights)
    if has_bias:
        getattr(module, name).bias = old_bias


def quantize_embedding(
    module: nn.Module | AutoModelForCausalLM,
    name: str,
    child: nn.Embedding,
    goal_dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
):
    old_weights = child.weight
    quantized_layer = Qint_Embedding(
        child.num_embeddings,
        child.embedding_dim,
        child.padding_idx,
        child.weight.dtype,
        goal_dtype,
        device,
        child.max_norm,
        child.norm_type,
        child.scale_grad_by_freq,
        child.sparse,
    )

    setattr(module, name, quantized_layer)

    getattr(module, name).quantize(old_weights)


def quantize_layernorm(
    module: nn.Module | AutoModelForCausalLM,
    name: str,
    child: nn.LayerNorm,
    goal_dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
):
    old_bias, old_weights = child.bias, child.weight
    quantized_layer = Qint_LayerNorm(
        child.normalized_shape,
        child.weight.dtype,
        goal_dtype,
        device,
        child.bias is not None,
    )

    setattr(module, name, quantized_layer)

    getattr(module, name).quantize(old_weights)
    getattr(module, name).bias = old_bias


# todo: clean this up, two calls to the same thing
def quantize_layers(
    module: AutoModelForCausalLM,
    goal_dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
    select_layers: list[str] | None = None,
):
    for name, child in module.named_children():  # type: ignore
        can_quantize = select_layers is None or name in select_layers
        if can_quantize and isinstance(child, nn.Linear):
            quantize_linear(module, name, child, goal_dtype, device)
        elif can_quantize and isinstance(child, nn.Embedding):
            quantize_embedding(module, name, child, goal_dtype, device)
        elif can_quantize and isinstance(child, nn.LayerNorm):
            quantize_layernorm(module, name, child, goal_dtype, device)
        else:
            quantize_layers(child, goal_dtype, device, select_layers)
