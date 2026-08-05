# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SM90 BF16 persistent grouped linear over external PyTorch tensors.

This module provides the training-facing autograd adapter for the CUTLASS
CuTeDSL grouped-GEMM kernel, including persistent grouped forward, dgrad,
wgrad, and optional direct ``main_grad`` accumulation.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Sequence

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch


_EXAMPLE = (
    Path(__file__).resolve().parent
    / "cutlass-v4.5.0/examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("param3_cutlass_grouped_gemm", _EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CuTeDSL grouped GEMM from {_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_grouped_gemm = _load_example_module()

# CuTeDSL specializes on tensor types/layouts, not the runtime values in the
# shape, pointer, or cluster-count tensors.  Param3 creates one adapter per MoE
# layer even though all of those adapters have the same specialization.  Keep
# the compiled callable process-local and let every layer retain independent
# metadata buffers.  This avoids repeating the expensive JIT compile without
# sharing mutable launch state between layers.
_COMPILED_FORWARD_CACHE = {}


class SM90BF16GroupedLinearRuntime:
    """Compiled forward runtime for a fixed number of BF16 expert weights."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        max_total_tokens: int,
        compile_splits: Sequence[int],
        grouped_backward: bool = False,
        direct_backward_layouts: bool = False,
        tile_shape_mn: tuple[int, int] = (128, 256),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        raster_along_m: bool = True,
        dgrad_tile_shape_mn: tuple[int, int] = (128, 256),
        dgrad_cluster_shape_mn: tuple[int, int] | None = None,
        dgrad_raster_along_m: bool = True,
        wgrad_tile_shape_mn: tuple[int, int] = (128, 256),
        wgrad_cluster_shape_mn: tuple[int, int] | None = None,
        wgrad_raster_along_m: bool | None = None,
        wgrad_accumulate_tile_shape_mn: tuple[int, int] = (128, 256),
    ) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError("SM90BF16GroupedLinearRuntime requires an H100-class SM90 GPU")
        if num_experts <= 0 or in_features <= 0 or out_features <= 0:
            raise ValueError("Grouped-linear dimensions must be positive")

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.max_total_tokens = max_total_tokens
        self.tile_shape_mn = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        if len(compile_splits) != num_experts or sum(compile_splits) > max_total_tokens:
            raise ValueError("compile_splits must describe one legal count per expert")

        dtype = cutlass.BFloat16
        acc_dtype = cutlass.Float32
        alignment_elements = 16 * 8 // dtype.width

        # Tiny tensors carry only dtype and major-mode information into JIT.
        self.initial_a = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, False, dtype
        )[2]
        self.initial_b = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, False, dtype
        )[2]
        self.initial_c = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, False, dtype
        )[2]

        placeholder_shapes = [(0, self.out_features, self.in_features, 1) for _ in range(self.num_experts)]
        placeholder_ptrs, fixed_strides = _grouped_gemm.create_group_metadata(placeholder_shapes, "k", "k", "n")
        self.shape_cute, self.shape_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor(placeholder_shapes, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.stride_cute, self.stride_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor(fixed_strides, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.ptr_cute, self.ptr_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor(placeholder_ptrs, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.shape_host = torch.empty((num_experts, 4), dtype=torch.int32, device="cpu", pin_memory=True)
        self.ptr_host = torch.empty((num_experts, 3), dtype=torch.int64, device="cpu", pin_memory=True)

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
        sm_count = hardware_info.get_max_active_clusters(1)
        tensormap_shape = (
            sm_count,
            _grouped_gemm.HopperGroupedGemmPersistentKernel.num_tensormaps,
            _grouped_gemm.HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8,
        )
        self.tensormap_cute, self.tensormap_tensor = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        # The local kernel variant reads this scalar inside the persistent
        # scheduler, so its value may change between launches without a re-JIT.
        cluster_tile_m = tile_shape_mn[0] * cluster_shape_mn[0]
        cluster_tile_n = tile_shape_mn[1] * cluster_shape_mn[1]
        max_m_tiles = sum(math.ceil(rows / cluster_tile_m) for rows in compile_splits)
        n_tiles = math.ceil(out_features / cluster_tile_n)
        max_total_clusters = max_m_tiles * n_tiles
        self.total_clusters_cute, self.total_clusters_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor([max_total_clusters], dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.total_clusters_host = torch.empty((1,), dtype=torch.int32, device="cpu", pin_memory=True)

        self.stream = cutlass_torch.default_stream()
        cache_key = (
            torch.cuda.current_device(),
            num_experts,
            in_features,
            out_features,
            tile_shape_mn,
            cluster_shape_mn,
            raster_along_m,
            max_active_clusters,
        )
        self.compiled = _COMPILED_FORWARD_CACHE.get(cache_key)
        if self.compiled is None:
            kernel = _grouped_gemm.HopperGroupedGemmPersistentKernel(
                acc_dtype,
                tile_shape_mn,
                cluster_shape_mn,
                swizzle_size=1,
                raster_along_m=raster_along_m,
                tensormap_update_mode=cutlass.utils.TensorMapUpdateMode.SMEM,
            )
            self.compiled = cute.compile(
                kernel,
                self.initial_a,
                self.initial_b,
                self.initial_c,
                num_experts,
                self.shape_cute,
                self.stride_cute,
                self.ptr_cute,
                self.total_clusters_cute,
                self.tensormap_cute,
                max_active_clusters,
                self.stream,
            )
            _COMPILED_FORWARD_CACHE[cache_key] = self.compiled

        self.backward_runtime = None
        if grouped_backward:
            # Import lazily after this module has finished defining the shared
            # grouped-GEMM source object used by the backward implementation.
            from param3_sm90_grouped_backward import (
                SM90BF16GroupedLinearBackwardRuntime,
            )

            self.backward_runtime = SM90BF16GroupedLinearBackwardRuntime(
                num_experts,
                in_features,
                out_features,
                max_tokens_per_expert=max_total_tokens,
                direct_layouts=direct_backward_layouts,
                dgrad_tile_shape_mn=dgrad_tile_shape_mn,
                dgrad_cluster_shape_mn=dgrad_cluster_shape_mn,
                dgrad_raster_along_m=dgrad_raster_along_m,
                wgrad_tile_shape_mn=wgrad_tile_shape_mn,
                wgrad_cluster_shape_mn=wgrad_cluster_shape_mn,
                wgrad_raster_along_m=wgrad_raster_along_m,
                wgrad_accumulate_tile_shape_mn=wgrad_accumulate_tile_shape_mn,
                fused_wgrad_accumulation=os.environ.get("MCORE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION", "0") == "1",
            )

    def __call__(
        self,
        inputs: torch.Tensor,
        weights: Sequence[torch.Tensor],
        splits: Sequence[int],
    ) -> torch.Tensor:
        if inputs.dtype != torch.bfloat16 or inputs.ndim != 2:
            raise TypeError("inputs must be a 2D BF16 tensor")
        if not inputs.is_cuda or not inputs.is_contiguous():
            raise ValueError("inputs must be contiguous CUDA storage")
        if inputs.shape[1] != self.in_features:
            raise ValueError(f"Expected input width {self.in_features}, got {inputs.shape[1]}")
        if len(weights) != self.num_experts or len(splits) != self.num_experts:
            raise ValueError("weights and splits must have one entry per local expert")
        if sum(splits) != inputs.shape[0] or sum(splits) > self.max_total_tokens:
            raise ValueError("Expert splits must cover inputs and remain within max_total_tokens")
        actual_total_clusters = sum(
            math.ceil(rows / (self.tile_shape_mn[0] * self.cluster_shape_mn[0])) for rows in splits
        ) * math.ceil(self.out_features / (self.tile_shape_mn[1] * self.cluster_shape_mn[1]))

        output = torch.empty((inputs.shape[0], self.out_features), dtype=torch.bfloat16, device=inputs.device)
        shapes = []
        ptrs = []
        offset = 0
        for expert, (weight, rows) in enumerate(zip(weights, splits)):
            if rows < 0:
                raise ValueError(f"Negative token count for expert {expert}")
            if (
                weight.dtype != torch.bfloat16
                or not weight.is_cuda
                or not weight.is_contiguous()
                or tuple(weight.shape) != (self.out_features, self.in_features)
            ):
                raise ValueError(
                    f"Expert {expert} weight must be contiguous CUDA BF16 [{self.out_features}, {self.in_features}]"
                )
            shapes.append((rows, self.out_features, self.in_features, 1))
            ptrs.append(
                [
                    inputs[offset:].data_ptr(),
                    weight.data_ptr(),
                    output[offset:].data_ptr(),
                ]
            )
            offset += rows

        self.shape_host.copy_(torch.tensor(shapes, dtype=torch.int32))
        self.ptr_host.copy_(torch.tensor(ptrs, dtype=torch.int64))
        self.total_clusters_host[0] = actual_total_clusters
        self.shape_tensor.copy_(self.shape_host, non_blocking=True)
        self.ptr_tensor.copy_(self.ptr_host, non_blocking=True)
        self.total_clusters_tensor.copy_(self.total_clusters_host, non_blocking=True)
        self.compiled(
            self.initial_a,
            self.initial_b,
            self.initial_c,
            self.shape_cute,
            self.stride_cute,
            self.ptr_cute,
            self.total_clusters_cute,
            self.tensormap_cute,
            cutlass_torch.current_stream(),
        )
        return output


class _SM90GroupedLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        splits,
        runtime,
        fuse_wgrad_accumulation,
        is_first_microbatch,
        *weights,
    ):
        splits = tuple(int(x) for x in splits)
        if runtime.backward_runtime is not None:
            runtime.backward_runtime.prepare_dgrad_weights(weights, is_first_microbatch)
        output = runtime(inputs, weights, splits)
        ctx.splits = splits
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        ctx.is_first_microbatch = is_first_microbatch
        ctx.runtime = runtime
        ctx.save_for_backward(inputs, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, *weights = ctx.saved_tensors
        direct_main_grad = False
        accumulate_main_grad = False
        main_grads = None
        if (
            ctx.fuse_wgrad_accumulation
            and ctx.runtime.backward_runtime is not None
            and ctx.runtime.backward_runtime.fused_wgrad_accumulation
        ):
            candidate_main_grads = [getattr(weight, "main_grad", None) for weight in weights]
            overwrite_flags = [bool(getattr(weight, "overwrite_main_grad", False)) for weight in weights]
            accumulation_flags = [
                ctx.is_first_microbatch is not True and not overwrite for overwrite in overwrite_flags
            ]
            direct_main_grad = (
                all(
                    grad is not None and grad.dtype == torch.bfloat16 and grad.is_cuda and grad.is_contiguous()
                    for grad in candidate_main_grads
                )
                and len(set(accumulation_flags)) == 1
            )
            if direct_main_grad:
                main_grads = candidate_main_grads
                accumulate_main_grad = accumulation_flags[0]
        if ctx.runtime.backward_runtime is not None:
            grad_input, grad_weights = ctx.runtime.backward_runtime(
                inputs,
                grad_output.contiguous(),
                weights,
                ctx.splits,
                main_grads=main_grads,
                accumulate_main_grad=accumulate_main_grad,
            )
        else:
            grad_input = torch.empty_like(inputs)
            grad_weights = []
            offset = 0
            for rows, weight in zip(ctx.splits, weights):
                x = inputs[offset : offset + rows]
                dy = grad_output[offset : offset + rows]
                if rows == 0:
                    grad_weights.append(torch.zeros_like(weight))
                else:
                    grad_input[offset : offset + rows] = dy @ weight
                    grad_weights.append(dy.transpose(0, 1) @ x)
                offset += rows
        if ctx.fuse_wgrad_accumulation:
            dummy_grads = []
            for weight, wgrad in zip(weights, grad_weights):
                if not hasattr(weight, "main_grad") or weight.main_grad is None:
                    raise RuntimeError("fuse_wgrad_accumulation requires a preallocated weight.main_grad")
                if not direct_main_grad:
                    overwrite = bool(getattr(weight, "overwrite_main_grad", False))
                    accumulate = ctx.is_first_microbatch is not True and not overwrite
                    if accumulate:
                        weight.main_grad.add_(wgrad.to(weight.main_grad.dtype))
                    else:
                        weight.main_grad.copy_(wgrad)
                if hasattr(weight, "grad_added_to_main_grad"):
                    weight.grad_added_to_main_grad = True
                dummy_grads.append(
                    torch.zeros_like(weight) if getattr(weight, "zero_out_wgrad", False) else torch.empty_like(weight)
                )
            grad_weights = dummy_grads
        return (grad_input, None, None, None, None, *grad_weights)


def sm90_grouped_linear(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    splits: Sequence[int],
    runtime: SM90BF16GroupedLinearRuntime,
    *,
    fuse_wgrad_accumulation: bool = False,
    is_first_microbatch: bool | None = None,
) -> torch.Tensor:
    """Run the experimental grouped linear with autograd-enabled weights."""

    return _SM90GroupedLinearAutograd.apply(
        inputs,
        tuple(splits),
        runtime,
        fuse_wgrad_accumulation,
        is_first_microbatch,
        *weights,
    )
