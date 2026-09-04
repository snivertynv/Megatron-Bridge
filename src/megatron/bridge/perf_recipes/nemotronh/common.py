# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: F401
"""Common helpers for nemotronh performance recipes."""

import os
from pathlib import Path

from megatron.core.quantization.utils import load_quantization_recipe

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotronh import nemotronh_56b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, nemotron_3_super_bf16_with_nvfp4_mixed


_TE_QUANT_CFG_PATH = Path(__file__).with_name("te_quant.cfg")


def _enable_mamba_true_cp_v3(cfg: ConfigContainer) -> None:
    """Select the opt-in Mamba True Context Parallel v3 training path."""
    if os.getenv("SOOFI_ENABLE_MAMBA_TRUE_CP_V3", "0") != "1":
        return

    from megatron.core.models.mamba.mamba_layer_specs_true_cp_v3 import mamba_stack_spec

    cfg.model.hybrid_stack_spec = mamba_stack_spec
    cfg.model.mamba_true_cp_chunk_size = int(
        os.getenv("SOOFI_MAMBA_TRUE_CP_CHUNK_SIZE", "256")
    )
    cfg.model.mamba_true_cp_bwd_subsequence_size = int(
        os.getenv("SOOFI_MAMBA_TRUE_CP_BWD_SUBSEQUENCE_SIZE", "64")
    )


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _nemotron_3_super_nvfp4_precision() -> MixedPrecisionConfig:
    """Return the NVFP4 precision config used by Nemotron 3 Super perf recipes."""
    cfg = nemotron_3_super_bf16_with_nvfp4_mixed()
    # Disabled until MCore PR 4358 lands.
    cfg.fp4_param_gather = False
    return cfg


def _apply_nemotron_3_super_perf_defaults(cfg: ConfigContainer) -> None:
    """Apply shared Nemotron 3 Super perf defaults after recipe-specific overrides."""
    _enable_mamba_true_cp_v3(cfg)

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.checkpoint.async_save = False

    _benchmark_common(cfg)
