# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from megatron.bridge.models.param3 import Param3ModelProvider


def test_param3_provider_preserves_mhc_fields() -> None:
    provider = Param3ModelProvider(num_layers=2, hidden_size=128, num_attention_heads=4)

    assert provider.enable_hyper_connections is True
    assert provider.use_fused_mhc is False
    assert provider.num_residual_streams == 4
    assert provider.mhc_sinkhorn_iterations == 20
    assert provider.transformer_layer_spec is get_transformer_block_with_experimental_attention_variant_spec
