# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Single-launch packed transpose for ragged expert-token matrices."""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_ragged_transpose_kernel(
    input_ptr,
    output_ptr,
    offsets_ptr,
    feature_size: tl.constexpr,
    block_rows: tl.constexpr,
    block_features: tl.constexpr,
):
    expert = tl.program_id(0)
    row_block = tl.program_id(1)
    feature_block = tl.program_id(2)

    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)
    rows = end - start
    row = row_block * block_rows + tl.arange(0, block_rows)[:, None]
    feature = feature_block * block_features + tl.arange(0, block_features)[None, :]
    mask = (row < rows) & (feature < feature_size)

    input_offset = (start + row) * feature_size + feature
    # Each expert occupies the same number of packed elements before and after
    # transpose, so start*feature_size is also its output base.
    output_offset = start * feature_size + feature * rows + row
    values = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, values, mask=mask)


def packed_ragged_transpose(
    inputs: torch.Tensor,
    offsets: torch.Tensor,
    splits: Sequence[int],
) -> torch.Tensor:
    """Transpose every packed expert matrix into one contiguous output buffer."""

    if inputs.ndim != 2 or not inputs.is_cuda or not inputs.is_contiguous():
        raise ValueError("inputs must be a contiguous 2D CUDA tensor")
    if offsets.dtype != torch.int32 or not offsets.is_cuda:
        raise ValueError("offsets must be a CUDA int32 tensor")
    if offsets.numel() != len(splits) + 1:
        raise ValueError("offsets must contain one prefix boundary per expert")

    output = torch.empty_like(inputs)
    max_rows = max(splits, default=0)
    if max_rows == 0:
        return output

    block_rows = 16
    block_features = 64
    grid = (
        len(splits),
        triton.cdiv(max_rows, block_rows),
        triton.cdiv(inputs.shape[1], block_features),
    )
    _packed_ragged_transpose_kernel[grid](
        inputs,
        output,
        offsets,
        feature_size=inputs.shape[1],
        block_rows=block_rows,
        block_features=block_features,
        num_warps=4,
    )
    return output
