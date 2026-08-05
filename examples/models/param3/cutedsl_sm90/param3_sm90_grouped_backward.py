# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SM90 BF16 persistent grouped-GEMM backward primitives for Param3 experts.

This module is kept separate from the live training adapter until its dgrad and
wgrad paths pass standalone numerical validation.  It reuses the same CUTLASS
4.5 Hopper grouped-GEMM kernel as the validated forward adapter.  The installed
CuTeDSL compiler does not finish transposed-major specializations in a practical
amount of time, so the first usable implementation materializes K-major operand
views before issuing one grouped dgrad and one grouped wgrad launch.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Sequence

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from param3_packed_transpose import packed_ragged_transpose
from param3_sm90_grouped_linear import _grouped_gemm


# As with the forward adapter, shape and pointer values are runtime data.  One
# compiled dgrad or wgrad specialization can therefore be reused by every MoE
# layer in a process while each runtime keeps independent metadata storage.
_COMPILED_BACKWARD_CACHE = {}
logger = logging.getLogger(__name__)


class _SM90BF16GroupedGemmRuntime:
    """One compiled persistent grouped GEMM with runtime shapes and pointers."""

    def __init__(
        self,
        compile_shapes: Sequence[tuple[int, int, int, int]],
        *,
        a_major: str,
        b_major: str,
        c_major: str,
        tile_shape_mn: tuple[int, int] = (128, 256),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        raster_along_m: bool = True,
        symbolic_initial_layouts: bool = False,
        runtime_strides: bool = True,
        accumulate_c: bool = False,
    ) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError("Grouped backward requires an H100-class SM90 GPU")
        if not compile_shapes:
            raise ValueError("compile_shapes must contain at least one expert")
        if a_major not in {"k", "m"} or b_major not in {"k", "n"}:
            raise ValueError("Unsupported A/B major mode")
        if c_major not in {"n", "m"}:
            raise ValueError("Unsupported C major mode")

        self.num_experts = len(compile_shapes)
        self.a_major = a_major
        self.b_major = b_major
        self.c_major = c_major
        self.tile_shape_mn = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        self.symbolic_initial_layouts = symbolic_initial_layouts
        self.runtime_strides = runtime_strides
        self.accumulate_c = accumulate_c

        dtype = cutlass.BFloat16
        alignment_elements = 16 * 8 // dtype.width
        self.initial_a = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, a_major == "m", dtype
        )[2]
        self.initial_b = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, b_major == "n", dtype
        )[2]
        self.initial_c = _grouped_gemm.create_tensor_and_stride(
            1, alignment_elements, alignment_elements, c_major == "m", dtype
        )[2]

        placeholder_ptrs, fixed_strides = _grouped_gemm.create_group_metadata(
            list(compile_shapes), a_major, b_major, c_major
        )
        stride_signature = tuple(tuple(stride) for stride in fixed_strides[0])
        self.shape_cute, self.shape_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor(compile_shapes, dtype=torch.int32),
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
        self.shape_host = torch.empty((self.num_experts, 4), dtype=torch.int32, device="cpu", pin_memory=True)
        self.ptr_host = torch.empty((self.num_experts, 3), dtype=torch.int64, device="cpu", pin_memory=True)
        self.stride_host = torch.empty(
            (self.num_experts, 3, 2),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
        sm_count = hardware_info.get_max_active_clusters(1)
        tensormap_shape = (
            sm_count,
            (_grouped_gemm.HopperGroupedGemmPersistentKernel.num_tensormaps + int(accumulate_c)),
            _grouped_gemm.HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8,
        )
        self.tensormap_cute, self.tensormap_tensor = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        max_total_clusters = sum(
            math.ceil(m / (tile_shape_mn[0] * cluster_shape_mn[0]))
            * math.ceil(n / (tile_shape_mn[1] * cluster_shape_mn[1]))
            for m, n, _, _ in compile_shapes
        )
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
            self.num_experts,
            a_major,
            b_major,
            c_major,
            stride_signature,
            tile_shape_mn,
            cluster_shape_mn,
            raster_along_m,
            symbolic_initial_layouts,
            accumulate_c,
            max_active_clusters,
        )
        self.compiled = _COMPILED_BACKWARD_CACHE.get(cache_key)
        if self.compiled is None:
            kernel = _grouped_gemm.HopperGroupedGemmPersistentKernel(
                cutlass.Float32,
                tile_shape_mn,
                cluster_shape_mn,
                swizzle_size=1,
                raster_along_m=raster_along_m,
                tensormap_update_mode=cutlass.utils.TensorMapUpdateMode.SMEM,
                accumulate_c=accumulate_c,
            )
            compile_initial_a = self.initial_a
            compile_initial_b = self.initial_b
            compile_initial_c = self.initial_c
            if symbolic_initial_layouts:

                def symbolic_carrier(mode0_major: bool):
                    shape = (
                        cute.sym_int32(divisibility=8),
                        cute.sym_int32(divisibility=8),
                        1,
                    )
                    return make_fake_compact_tensor(
                        dtype,
                        shape,
                        stride_order=(0, 1, 2) if mode0_major else (1, 0, 2),
                        assumed_align=16,
                    )

                compile_initial_a = symbolic_carrier(a_major == "m")
                compile_initial_b = symbolic_carrier(b_major == "n")
                compile_initial_c = symbolic_carrier(c_major == "m")
            self.compiled = cute.compile(
                kernel,
                compile_initial_a,
                compile_initial_b,
                compile_initial_c,
                self.num_experts,
                self.shape_cute,
                self.stride_cute,
                self.ptr_cute,
                self.total_clusters_cute,
                self.tensormap_cute,
                max_active_clusters,
                self.stream,
            )
            _COMPILED_BACKWARD_CACHE[cache_key] = self.compiled

        # Symbolic compilation needs a runtime tensor with a dynamic layout,
        # but the grouped metadata rewrites every descriptor field before the
        # operand is used. Keep three tiny, correctly-strided carriers instead
        # of rebuilding six DLPack objects for every direct dgrad/wgrad call.
        self.runtime_initial = None
        self.runtime_initial_tensors = None
        if symbolic_initial_layouts:
            runtime_initial = []
            runtime_initial_tensors = []
            for mode0_major in (
                a_major == "m",
                b_major == "n",
                c_major == "m",
            ):
                base = torch.empty(
                    (alignment_elements, alignment_elements),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                tensor = base.transpose(0, 1) if mode0_major else base
                carrier = from_dlpack(tensor, assumed_align=16)
                carrier.element_type = cutlass.BFloat16
                carrier = carrier.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))
                runtime_initial_tensors.append(tensor)
                runtime_initial.append(carrier)
            self.runtime_initial_tensors = tuple(runtime_initial_tensors)
            self.runtime_initial = tuple(runtime_initial)

    def __call__(
        self,
        shapes: Sequence[tuple[int, int, int, int]],
        ptrs: Sequence[tuple[int, int, int]],
        initial_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        if len(shapes) != self.num_experts or len(ptrs) != self.num_experts:
            raise ValueError("shapes and ptrs must contain one entry per expert")
        total_clusters = sum(
            math.ceil(m / (self.tile_shape_mn[0] * self.cluster_shape_mn[0]))
            * math.ceil(n / (self.tile_shape_mn[1] * self.cluster_shape_mn[1]))
            for m, n, _, _ in shapes
        )
        self.shape_host.copy_(torch.tensor(shapes, dtype=torch.int32))
        self.ptr_host.copy_(torch.tensor(ptrs, dtype=torch.int64))
        self.total_clusters_host[0] = total_clusters
        self.shape_tensor.copy_(self.shape_host, non_blocking=True)
        self.ptr_tensor.copy_(self.ptr_host, non_blocking=True)
        if self.runtime_strides:
            _, strides = _grouped_gemm.create_group_metadata(list(shapes), self.a_major, self.b_major, self.c_major)
            self.stride_host.copy_(torch.tensor(strides, dtype=torch.int32))
            self.stride_tensor.copy_(self.stride_host, non_blocking=True)
        self.total_clusters_tensor.copy_(self.total_clusters_host, non_blocking=True)
        runtime_initial = (self.initial_a, self.initial_b, self.initial_c)
        if self.symbolic_initial_layouts:
            runtime_initial = self.runtime_initial
        self.compiled(
            runtime_initial[0],
            runtime_initial[1],
            runtime_initial[2],
            self.shape_cute,
            self.stride_cute,
            self.ptr_cute,
            self.total_clusters_cute,
            self.tensormap_cute,
            cutlass_torch.current_stream(),
        )


class SM90BF16GroupedLinearBackwardRuntime:
    """Persistent grouped dgrad and wgrad for a fixed expert weight shape."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        max_tokens_per_expert: int,
        direct_layouts: bool = False,
        dgrad_tile_shape_mn: tuple[int, int] = (128, 256),
        dgrad_cluster_shape_mn: tuple[int, int] | None = None,
        dgrad_raster_along_m: bool = True,
        wgrad_tile_shape_mn: tuple[int, int] = (128, 256),
        wgrad_cluster_shape_mn: tuple[int, int] | None = None,
        wgrad_raster_along_m: bool | None = None,
        wgrad_accumulate_tile_shape_mn: tuple[int, int] = (128, 256),
        fused_wgrad_accumulation: bool = False,
    ) -> None:
        if min(num_experts, in_features, out_features, max_tokens_per_expert) <= 0:
            raise ValueError("Backward runtime dimensions must be positive")
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.max_tokens_per_expert = max_tokens_per_expert
        self.direct_layouts = direct_layouts
        self.fused_wgrad_accumulation = fused_wgrad_accumulation
        if fused_wgrad_accumulation and not direct_layouts:
            raise ValueError("Fused wgrad accumulation requires direct layouts")
        if dgrad_cluster_shape_mn is None:
            dgrad_cluster_shape_mn = (2, 1) if direct_layouts else (1, 1)
        if wgrad_cluster_shape_mn is None:
            wgrad_cluster_shape_mn = (2, 1) if direct_layouts else (1, 1)
        if wgrad_raster_along_m is None:
            wgrad_raster_along_m = not direct_layouts
        self.offsets_host = torch.empty((num_experts + 1,), dtype=torch.int32, device="cpu", pin_memory=True)
        self.offsets_device = torch.empty((num_experts + 1,), dtype=torch.int32, device="cuda")
        self._dgrad_weights = None
        compile_tokens = min(max_tokens_per_expert, 128)

        # dX = dY @ W. Direct mode consumes the physical [out, in] weight as a
        # logical N-major [in, out] view; compatibility mode materializes K-major.
        self.dgrad = _SM90BF16GroupedGemmRuntime(
            [(compile_tokens, in_features, out_features, 1)] * num_experts,
            a_major="k",
            b_major="n" if direct_layouts else "k",
            c_major="n",
            tile_shape_mn=dgrad_tile_shape_mn,
            cluster_shape_mn=dgrad_cluster_shape_mn,
            raster_along_m=dgrad_raster_along_m,
            symbolic_initial_layouts=direct_layouts,
            runtime_strides=not direct_layouts,
        )
        # dW = dY^T @ X. Direct mode describes both transpose views through
        # strides; compatibility mode materializes two packed K-major operands.
        self.wgrad = _SM90BF16GroupedGemmRuntime(
            [(out_features, in_features, compile_tokens, 1)] * num_experts,
            a_major="m" if direct_layouts else "k",
            b_major="n" if direct_layouts else "k",
            c_major="n",
            tile_shape_mn=wgrad_tile_shape_mn,
            cluster_shape_mn=wgrad_cluster_shape_mn,
            raster_along_m=wgrad_raster_along_m,
            symbolic_initial_layouts=direct_layouts,
            runtime_strides=not direct_layouts,
        )
        self.wgrad_accumulate = None
        if fused_wgrad_accumulation:
            self.wgrad_accumulate = _SM90BF16GroupedGemmRuntime(
                [(out_features, in_features, compile_tokens, 1)] * num_experts,
                a_major="m",
                b_major="n",
                c_major="n",
                tile_shape_mn=wgrad_accumulate_tile_shape_mn,
                # The beta epilogue is validated with the non-multicast
                # geometry. Keep overwrite wgrad on its faster 2x1/N tuning.
                cluster_shape_mn=(1, 1),
                raster_along_m=True,
                symbolic_initial_layouts=True,
                runtime_strides=False,
                accumulate_c=True,
            )

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor, shape: tuple[int, int], name: str) -> None:
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{name} must be contiguous CUDA BF16 with shape {shape}")

    def prepare_dgrad_weights(
        self,
        weights: Sequence[torch.Tensor],
        is_first_microbatch: bool | None,
    ) -> None:
        """Refresh the packed transposed weights once per optimizer step."""

        if self.direct_layouts:
            self._dgrad_weights = None
            return
        if is_first_microbatch is None:
            # Without TE's step-boundary signal, retaining a transpose across
            # optimizer updates could use stale parameters. Fall back to the
            # per-call path in __call__.
            self._dgrad_weights = None
            return
        if is_first_microbatch or self._dgrad_weights is None:
            self._dgrad_weights = torch.stack([weight.transpose(0, 1) for weight in weights], dim=0)

    def __call__(
        self,
        inputs: torch.Tensor,
        grad_output: torch.Tensor,
        weights: Sequence[torch.Tensor],
        splits: Sequence[int],
        *,
        main_grads: Sequence[torch.Tensor] | None = None,
        accumulate_main_grad: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        total_tokens = sum(splits)
        self._validate_tensor(inputs, (total_tokens, self.in_features), "inputs")
        self._validate_tensor(grad_output, (total_tokens, self.out_features), "grad_output")
        if len(weights) != self.num_experts or len(splits) != self.num_experts:
            raise ValueError("weights and splits must contain one entry per expert")
        if any(rows < 0 or rows > self.max_tokens_per_expert for rows in splits):
            raise ValueError("Expert split exceeds compiled backward capacity")
        for expert, weight in enumerate(weights):
            self._validate_tensor(weight, (self.out_features, self.in_features), f"weight[{expert}]")

        offsets = [0]
        for rows in splits:
            offsets.append(offsets[-1] + rows)
        if not self.direct_layouts:
            self.offsets_host.copy_(torch.tensor(offsets, dtype=torch.int32))
            self.offsets_device.copy_(self.offsets_host, non_blocking=True)

        grad_input = torch.empty_like(inputs)
        if main_grads is not None:
            if not self.fused_wgrad_accumulation or self.wgrad_accumulate is None:
                raise ValueError("Direct main_grad output was not enabled for this runtime")
            if len(main_grads) != self.num_experts:
                raise ValueError("main_grads must contain one tensor per expert")
            grad_weights = list(main_grads)
            for expert, grad_weight in enumerate(grad_weights):
                self._validate_tensor(
                    grad_weight,
                    (self.out_features, self.in_features),
                    f"main_grad[{expert}]",
                )
        else:
            grad_weights_buffer = torch.empty(
                (self.num_experts, self.out_features, self.in_features),
                dtype=torch.bfloat16,
                device=inputs.device,
            )
            grad_weights = [grad_weights_buffer[expert] for expert in range(self.num_experts)]
        # A non-empty grouped wgrad overwrites its entire expert matrix. Only a
        # zero-token expert has no tiles and therefore needs explicit zeroing.
        for rows, grad_weight in zip(splits, grad_weights):
            if rows == 0 and not accumulate_main_grad:
                grad_weight.zero_()
        dgrad_weights = weights if self.direct_layouts else self._dgrad_weights
        if dgrad_weights is None:
            # Correctness fallback for callers without TE's first-microbatch
            # signal. Training prepares and reuses this tensor in forward.
            dgrad_weights = torch.stack([weight.transpose(0, 1) for weight in weights], dim=0)
        wgrad_a_buffer = None
        wgrad_b_buffer = None
        if not self.direct_layouts:
            wgrad_a_buffer = packed_ragged_transpose(grad_output, self.offsets_device, splits)
            wgrad_b_buffer = packed_ragged_transpose(inputs, self.offsets_device, splits)
        dgrad_shapes = []
        dgrad_ptrs = []
        wgrad_shapes = []
        wgrad_ptrs = []
        offset = 0
        element_size = inputs.element_size()
        for rows, dgrad_weight, grad_weight in zip(splits, dgrad_weights, grad_weights):
            dgrad_shapes.append((rows, self.in_features, self.out_features, 1))
            dgrad_ptrs.append(
                (
                    grad_output[offset:].data_ptr(),
                    dgrad_weight.data_ptr(),
                    grad_input[offset:].data_ptr(),
                )
            )
            wgrad_shapes.append((self.out_features, self.in_features, rows, 1))
            if self.direct_layouts:
                wgrad_ptrs.append(
                    (
                        grad_output.data_ptr() + offset * self.out_features * element_size,
                        inputs.data_ptr() + offset * self.in_features * element_size,
                        grad_weight.data_ptr(),
                    )
                )
            else:
                wgrad_ptrs.append(
                    (
                        wgrad_a_buffer.data_ptr() + offset * self.out_features * element_size,
                        wgrad_b_buffer.data_ptr() + offset * self.in_features * element_size,
                        grad_weight.data_ptr(),
                    )
                )
            offset += rows

        if self.direct_layouts:
            representative = next(expert for expert, rows in enumerate(splits) if rows > 0)
            start = offsets[representative]
            stop = start + splits[representative]
            dy = grad_output[start:stop]
            x = inputs[start:stop]
            dx = grad_input[start:stop]
            self.dgrad(
                dgrad_shapes,
                dgrad_ptrs,
                (dy, weights[representative].transpose(0, 1), dx),
            )
            wgrad_runtime = self.wgrad_accumulate if main_grads is not None and accumulate_main_grad else self.wgrad
            if os.environ.get("MCORE_SM90_CUTEDSL_DEBUG_WGRAD", "0") == "1":
                logger.info(
                    "PARAM3_SM90_WGRAD_RUNTIME_BEGIN "
                    f"direct_main_grad={main_grads is not None} "
                    f"accumulate={accumulate_main_grad}"
                )
            wgrad_runtime(
                wgrad_shapes,
                wgrad_ptrs,
                (
                    dy.transpose(0, 1),
                    x.transpose(0, 1),
                    grad_weights[representative],
                ),
            )
            if os.environ.get("MCORE_SM90_CUTEDSL_DEBUG_WGRAD", "0") == "1":
                logger.info("PARAM3_SM90_WGRAD_RUNTIME_END")
        else:
            self.dgrad(dgrad_shapes, dgrad_ptrs)
            self.wgrad(wgrad_shapes, wgrad_ptrs)
        return grad_input, grad_weights
