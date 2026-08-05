# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: D103

"""Numerically validate persistent grouped dgrad/wgrad at Param3 FC1 shape."""

from __future__ import annotations

import argparse

import torch
from param3_sm90_grouped_backward import SM90BF16GroupedLinearBackwardRuntime


def reference_backward(inputs, grad_output, weights, splits):
    grad_input = torch.empty_like(inputs)
    grad_weights = []
    offset = 0
    for rows, weight in zip(splits, weights):
        x = inputs[offset : offset + rows]
        dy = grad_output[offset : offset + rows]
        if rows == 0:
            grad_weights.append(torch.zeros_like(weight))
        else:
            grad_input[offset : offset + rows] = dy @ weight
            grad_weights.append(dy.transpose(0, 1) @ x)
        offset += rows
    return grad_input, grad_weights


def validate_case(runtime, weights, splits, name):
    total_tokens = sum(splits)
    inputs = torch.randn(total_tokens, runtime.in_features, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(total_tokens, runtime.out_features, device="cuda", dtype=torch.bfloat16)
    expected_dx, expected_dw = reference_backward(inputs, grad_output, weights, splits)
    actual_dx, actual_dw = runtime(inputs, grad_output, weights, splits)
    torch.cuda.synchronize()

    dx_error = (actual_dx.float() - expected_dx.float()).abs().max().item()
    dw_error = max(
        (actual.float() - expected.float()).abs().max().item() for actual, expected in zip(actual_dw, expected_dw)
    )
    torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-2, atol=2.5e-1)
    for actual, expected in zip(actual_dw, expected_dw):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)
    print(
        f"PARAM3_SM90_GROUPED_BACKWARD_{name}_PASS dx_max_error={dx_error:.6f} dw_max_error={dw_error:.6f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-layouts", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    num_experts = 16
    in_features = 3072
    out_features = 3072
    runtime = SM90BF16GroupedLinearBackwardRuntime(
        num_experts,
        in_features,
        out_features,
        max_tokens_per_expert=128,
        direct_layouts=args.direct_layouts,
    )
    weights = [torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) for _ in range(num_experts)]
    runtime.prepare_dgrad_weights(weights, True)
    validate_case(runtime, weights, [64] * num_experts, "BALANCED")
    validate_case(
        runtime,
        weights,
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120],
        "RAGGED_ZERO_EXPERT",
    )


if __name__ == "__main__":
    main()
