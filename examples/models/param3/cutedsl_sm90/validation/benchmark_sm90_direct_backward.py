# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: D103

"""Benchmark transpose-free SM90 grouped dgrad and wgrad for Param3."""

from __future__ import annotations

import argparse

import torch
from param3_sm90_grouped_backward import _SM90BF16GroupedGemmRuntime
from param3_sm90_grouped_linear import SM90BF16GroupedLinearRuntime


def _parse_pairs(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(";"):
        m, n = item.lower().replace(",", "x").split("x")
        result.append((int(m), int(n)))
    return result


def _time_runtime(
    runtime,
    shapes,
    ptrs,
    initial_tensors,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        runtime(shapes, ptrs, initial_tensors)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        runtime(shapes, ptrs, initial_tensors)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _time_forward(
    runtime,
    inputs,
    weights,
    splits,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        runtime(inputs, weights, splits)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        runtime(inputs, weights, splits)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-expert", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--tiles", default="128x256")
    parser.add_argument("--clusters", default="1x1;1x2;2x1;2x2")
    parser.add_argument("--rasters", default="m;n")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--accumulate-c", action="store_true")
    parser.add_argument("--separate-outputs", action="store_true")
    parser.add_argument("--overwrite-before-accumulate", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    num_experts = args.num_experts
    in_features = args.in_features
    out_features = args.out_features
    rows = args.tokens_per_expert
    total_rows = num_experts * rows
    dtype = torch.bfloat16
    x = torch.randn(total_rows, in_features, dtype=dtype, device="cuda")
    dy = torch.randn(total_rows, out_features, dtype=dtype, device="cuda")
    weights = torch.randn(num_experts, out_features, in_features, dtype=dtype, device="cuda")
    dx = torch.empty_like(x)
    if args.separate_outputs:
        dw_experts = [torch.empty(out_features, in_features, dtype=dtype, device="cuda") for _ in range(num_experts)]
    else:
        dw = torch.empty_like(weights)
        dw_experts = [dw[expert] for expert in range(num_experts)]
    element_size = x.element_size()

    dgrad_shapes = [(rows, in_features, out_features, 1)] * num_experts
    wgrad_shapes = [(out_features, in_features, rows, 1)] * num_experts
    dgrad_ptrs = []
    wgrad_ptrs = []
    for expert in range(num_experts):
        offset = expert * rows
        dgrad_ptrs.append(
            (
                dy.data_ptr() + offset * out_features * element_size,
                weights[expert].data_ptr(),
                dx.data_ptr() + offset * in_features * element_size,
            )
        )
        wgrad_ptrs.append(
            (
                dy.data_ptr() + offset * out_features * element_size,
                x.data_ptr() + offset * in_features * element_size,
                dw_experts[expert].data_ptr(),
            )
        )

    dgrad_initial = (dy[:rows], weights[0].transpose(0, 1), dx[:rows])
    wgrad_initial = (
        dy[:rows].transpose(0, 1),
        x[:rows].transpose(0, 1),
        dw_experts[0],
    )
    for tile in _parse_pairs(args.tiles):
        for cluster in _parse_pairs(args.clusters):
            for raster in args.rasters.split(";"):
                raster = raster.strip().lower()
                if raster not in {"m", "n"}:
                    raise ValueError(f"Unsupported raster direction: {raster}")
                forward = SM90BF16GroupedLinearRuntime(
                    num_experts,
                    in_features,
                    out_features,
                    max_total_tokens=total_rows,
                    compile_splits=[rows] * num_experts,
                    tile_shape_mn=tile,
                    cluster_shape_mn=cluster,
                    raster_along_m=raster == "m",
                )
                forward_us = _time_forward(
                    forward,
                    x,
                    list(weights),
                    [rows] * num_experts,
                    args.warmups,
                    args.iterations,
                )
                dgrad = _SM90BF16GroupedGemmRuntime(
                    dgrad_shapes,
                    a_major="k",
                    b_major="n",
                    c_major="n",
                    tile_shape_mn=tile,
                    cluster_shape_mn=cluster,
                    raster_along_m=raster == "m",
                    symbolic_initial_layouts=True,
                )
                dgrad_us = _time_runtime(
                    dgrad,
                    dgrad_shapes,
                    dgrad_ptrs,
                    dgrad_initial,
                    args.warmups,
                    args.iterations,
                )
                wgrad = _SM90BF16GroupedGemmRuntime(
                    wgrad_shapes,
                    a_major="m",
                    b_major="n",
                    c_major="n",
                    tile_shape_mn=tile,
                    cluster_shape_mn=cluster,
                    raster_along_m=raster == "m",
                    symbolic_initial_layouts=True,
                    runtime_strides=False,
                    accumulate_c=args.accumulate_c,
                )
                if args.accumulate_c:
                    if args.overwrite_before_accumulate:
                        overwrite_wgrad = _SM90BF16GroupedGemmRuntime(
                            wgrad_shapes,
                            a_major="m",
                            b_major="n",
                            c_major="n",
                            tile_shape_mn=tile,
                            cluster_shape_mn=cluster,
                            raster_along_m=raster == "m",
                            symbolic_initial_layouts=True,
                            runtime_strides=False,
                        )
                        overwrite_wgrad(wgrad_shapes, wgrad_ptrs, wgrad_initial)
                        torch.cuda.synchronize()
                        dw_seed = [output.clone() for output in dw_experts]
                    else:
                        dw_seed = [torch.randn_like(output) for output in dw_experts]
                        for output, seed in zip(dw_experts, dw_seed):
                            output.copy_(seed)
                    wgrad(wgrad_shapes, wgrad_ptrs, wgrad_initial)
                    torch.cuda.synchronize()
                    gemm_ref = dy[:rows].float().transpose(0, 1) @ x[:rows].float()
                    ref = (gemm_ref + dw_seed[0].float()).to(dtype)
                    max_abs = float((dw_experts[0].float() - ref.float()).abs().max())
                    no_beta_max_abs = float((dw_experts[0].float() - gemm_ref.to(dtype).float()).abs().max())
                    recovered_beta_max_abs = float(
                        ((dw_experts[0].float() - gemm_ref.to(dtype).float()) - dw_seed[0].float()).abs().max()
                    )
                    print(
                        "PARAM3_SM90_ACCUMULATING_WGRAD_DIAGNOSTIC "
                        f"fused_max_abs={max_abs:.6f} "
                        f"no_beta_max_abs={no_beta_max_abs:.6f} "
                        f"recovered_beta_max_abs={recovered_beta_max_abs:.6f}",
                        flush=True,
                    )
                    if max_abs > 1.0:
                        raise AssertionError(f"Accumulating wgrad maximum error {max_abs} exceeds 1.0")
                    print(
                        f"PARAM3_SM90_ACCUMULATING_WGRAD_VALIDATION max_abs={max_abs:.6f} PASS",
                        flush=True,
                    )
                wgrad_us = _time_runtime(
                    wgrad,
                    wgrad_shapes,
                    wgrad_ptrs,
                    wgrad_initial,
                    args.warmups,
                    args.iterations,
                )
                print(
                    "PARAM3_SM90_DIRECT_BACKWARD_RESULT "
                    f"in_features={in_features} out_features={out_features} "
                    f"tile={tile[0]}x{tile[1]} "
                    f"cluster={cluster[0]}x{cluster[1]} raster={raster} "
                    f"forward_us={forward_us:.3f} "
                    f"dgrad_us={dgrad_us:.3f} wgrad_us={wgrad_us:.3f} "
                    f"three_gemm_us={forward_us + dgrad_us + wgrad_us:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
