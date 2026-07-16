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

"""Flat Param3 performance recipes for ``setup_experiment.py``."""

from megatron.bridge.recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config as _pp2_perf_config,
)
from megatron.bridge.recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config as _pp2_vp4_perf_config,
)
from megatron.bridge.recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config as _pp4_perf_config,
)
from megatron.bridge.recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_no_recompute_precision_aware_perf_config as _pp4_vp2_no_recompute_perf_config,
)
from megatron.bridge.recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config as _pp4_vp2_perf_config,
)
from megatron.bridge.training.config import ConfigContainer


def param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config() -> ConfigContainer:
    """Return the 32-GPU PP2 Param3 throughput recipe."""
    return _pp2_perf_config()


def param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config() -> ConfigContainer:
    """Return the 32-GPU interleaved PP2/VP4 Param3 recipe."""
    return _pp2_vp4_perf_config()


def param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config() -> ConfigContainer:
    """Return the 32-GPU PP4 Param3 throughput recipe."""
    return _pp4_perf_config()


def param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config() -> ConfigContainer:
    """Return the 32-GPU interleaved PP4/VP2 Param3 recipe."""
    return _pp4_vp2_perf_config()


def param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_no_recompute_precision_aware_perf_config() -> ConfigContainer:
    """Return the 32-GPU interleaved PP4/VP2 no-recompute probe."""
    return _pp4_vp2_no_recompute_perf_config()


__all__ = [
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_no_recompute_precision_aware_perf_config",
]
