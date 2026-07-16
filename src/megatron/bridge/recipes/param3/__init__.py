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

from megatron.bridge.recipes.param3.h100.param3 import (
    PARAM3_74B_ATTENTION_PATTERN,
    PARAM3_74B_MOE_PATTERN,
    param3_74b_pretrain_32gpu_h100_bf16_config,
    param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config,
    param3_74b_pretrain_32gpu_h100_bf16_fsdp_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp2_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp2_no_moe_recompute_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_no_recompute_precision_aware_perf_config,
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config,
)


__all__ = [
    "PARAM3_74B_ATTENTION_PATTERN",
    "PARAM3_74B_MOE_PATTERN",
    "param3_74b_pretrain_32gpu_h100_bf16_config",
    "param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config",
    "param3_74b_pretrain_32gpu_h100_bf16_fsdp_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_no_moe_recompute_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_no_recompute_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config",
]
