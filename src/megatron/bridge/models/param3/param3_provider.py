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

from dataclasses import dataclass
from typing import Callable

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class Param3ModelProvider(GPTModelProvider):
    """Provider for Param3 models using mHC and experimental attention.

    Param3 needs the experimental-attention block builder for its per-layer
    GatedDeltaNet/full-attention pattern. The mHC fields are declared here so
    they are retained in serialized Bridge run configurations when using a
    Megatron-Core revision that implements hyper-connections.
    """

    enable_hyper_connections: bool = True
    use_fused_mhc: bool = False
    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    transformer_layer_spec: Callable = get_transformer_block_with_experimental_attention_variant_spec
