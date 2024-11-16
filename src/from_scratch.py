from transformers import AutoModelForCausalLM

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def quantize_linear_layers(
    module: AutoModelForCausalLM,
    goal_dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
):
    for name, child in module.named_children():  # type: ignore
        if not isinstance(child, nn.Linear):
            quantize_linear_layers(child, goal_dtype, device)
            continue

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
