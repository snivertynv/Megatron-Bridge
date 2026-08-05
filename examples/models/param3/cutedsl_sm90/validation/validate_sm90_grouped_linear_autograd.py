#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: D103

"""Validate experimental CuTeDSL expert linear forward/backward against TE."""

from __future__ import annotations

import argparse

import torch
import transformer_engine.pytorch as te
from param3_sm90_grouped_linear import SM90BF16GroupedLinearRuntime, sm90_grouped_linear


NUM_EXPERTS = 16
IN_FEATURES = 3072
OUT_FEATURES = 3072
TOTAL_TOKENS = 4096


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max())


def run_case(runtime, layer, splits, name):
    torch.manual_seed(1234)
    x_base = torch.randn(TOTAL_TOKENS, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(TOTAL_TOKENS, OUT_FEATURES, dtype=torch.bfloat16, device="cuda")
    weights = [getattr(layer, f"weight{i}") for i in range(NUM_EXPERTS)]

    for weight in weights:
        weight.grad = None
    x_te = x_base.detach().clone().requires_grad_(True)
    y_te = layer(x_te, list(splits))
    y_te.backward(grad_output)
    dx_te = x_te.grad.detach().clone()
    dw_te = [weight.grad.detach().clone() for weight in weights]

    for weight in weights:
        weight.grad = None
    x_cute = x_base.detach().clone().requires_grad_(True)
    y_cute = sm90_grouped_linear(x_cute, weights, splits, runtime)
    y_cute.backward(grad_output)
    dx_cute = x_cute.grad.detach().clone()
    dw_cute = [weight.grad.detach().clone() for weight in weights]
    torch.cuda.synchronize()

    output_max = max_abs(y_te, y_cute)
    dinput_max = max_abs(dx_te, dx_cute)
    dweight_max = max(max_abs(a, b) for a, b in zip(dw_te, dw_cute))
    print(
        f"CASE {name} output_max_abs={output_max:.8f} "
        f"dinput_max_abs={dinput_max:.8f} dweight_max_abs={dweight_max:.8f}",
        flush=True,
    )
    torch.testing.assert_close(y_cute, y_te, atol=0.125, rtol=0.05)
    torch.testing.assert_close(dx_cute, dx_te, atol=0.25, rtol=0.05)
    for expert, (actual, expected) in enumerate(zip(dw_cute, dw_te)):
        try:
            torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.05)
        except AssertionError as error:
            raise AssertionError(f"Weight-gradient mismatch for expert {expert}: {error}")
    print(f"CASE {name} PASS", flush=True)


def run_main_grad_case(runtime, layer, splits):
    print("CASE main_grad setup", flush=True)
    torch.manual_seed(2468)
    inputs = torch.randn(TOTAL_TOKENS, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(TOTAL_TOKENS, OUT_FEATURES, dtype=torch.bfloat16, device="cuda")
    weights = [getattr(layer, f"weight{i}") for i in range(NUM_EXPERTS)]
    expected = []
    offset = 0
    for rows in splits:
        expected.append(grad_output[offset : offset + rows].transpose(0, 1) @ inputs[offset : offset + rows])
        offset += rows
    for weight in weights:
        weight.grad = None
        weight.main_grad = torch.zeros_like(weight)
        weight.grad_added_to_main_grad = False

    output = sm90_grouped_linear(
        inputs,
        weights,
        splits,
        runtime,
        fuse_wgrad_accumulation=True,
        is_first_microbatch=True,
    )
    print("CASE main_grad first backward begin", flush=True)
    output.backward(grad_output)
    print("CASE main_grad first backward end", flush=True)
    for expert, (weight, ref) in enumerate(zip(weights, expected)):
        torch.testing.assert_close(weight.main_grad, ref, atol=0.5, rtol=0.05)
        assert weight.grad_added_to_main_grad, expert
        weight.grad = None

    output = sm90_grouped_linear(
        inputs,
        weights,
        splits,
        runtime,
        fuse_wgrad_accumulation=True,
        is_first_microbatch=False,
    )
    print("CASE main_grad accumulate backward begin", flush=True)
    output.backward(grad_output)
    print("CASE main_grad accumulate backward end", flush=True)
    for expert, (weight, ref) in enumerate(zip(weights, expected)):
        torch.testing.assert_close(weight.main_grad, ref * 2, atol=1.0, rtol=0.05)
        assert weight.grad_added_to_main_grad, expert
    print("CASE main_grad_first_and_accumulate PASS", flush=True)


def main():
    global NUM_EXPERTS, IN_FEATURES, OUT_FEATURES, TOTAL_TOKENS

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned-direct-backward", action="store_true")
    parser.add_argument("--main-grad-only", action="store_true")
    parser.add_argument("--one-expert", action="store_true")
    parser.add_argument(
        "--fc2-shape",
        action="store_true",
        help="Use Param3 expert FC2 dimensions (1536 -> 3072).",
    )
    args = parser.parse_args()
    if args.fc2_shape:
        IN_FEATURES = 1536
        OUT_FEATURES = 3072
    if args.one_expert:
        NUM_EXPERTS = 1
        TOTAL_TOKENS = 256

    assert torch.cuda.get_device_capability()[0] == 9
    layer = te.GroupedLinear(
        NUM_EXPERTS,
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        return_bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        parallel_mode=None,
    )
    balanced = [TOTAL_TOKENS // NUM_EXPERTS] * NUM_EXPERTS
    ragged = [0, 1, 7, 15, 31, 63, 127, 255, 384, 512, 640, 768, 512, 384, 256, 141]
    runtime = SM90BF16GroupedLinearRuntime(
        NUM_EXPERTS,
        IN_FEATURES,
        OUT_FEATURES,
        max_total_tokens=TOTAL_TOKENS,
        compile_splits=balanced,
        grouped_backward=args.tuned_direct_backward,
        direct_backward_layouts=args.tuned_direct_backward,
        cluster_shape_mn=(2, 1) if args.tuned_direct_backward else (1, 1),
    )
    if not args.main_grad_only:
        run_case(runtime, layer, balanced, "balanced")
        run_case(runtime, layer, ragged, "ragged_zero_dynamic_scheduler")
    run_main_grad_case(runtime, layer, balanced)
    layer_name = "FC2" if args.fc2_shape else "FC1"
    print(f"AUTOGRAD_VALIDATION {layer_name} PASS", flush=True)


if __name__ == "__main__":
    main()
