# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from megatron.bridge.recipes.param3 import (
    PARAM3_74B_ATTENTION_PATTERN,
    PARAM3_74B_MOE_PATTERN,
    param3_74b_pretrain_32gpu_h100_bf16_config,
)


def test_param3_74b_pretrain_recipe_matches_logged_architecture() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_config()

    assert cfg.model.num_layers == 40
    assert cfg.model.hidden_size == 3072
    assert cfg.model.experimental_attention_variant == "gated_delta_net"
    assert cfg.model.linear_attention_freq == PARAM3_74B_ATTENTION_PATTERN
    assert len(cfg.model.linear_attention_freq) == cfg.model.num_layers
    assert cfg.model.linear_attention_freq.count(1) == 26
    assert cfg.model.linear_attention_freq.count(0) == 14
    assert cfg.model.enable_hyper_connections is True
    assert cfg.model.recompute_modules == ["layernorm", "moe_act", "moe", "mhc"]

    assert cfg.model.num_moe_experts == 128
    assert cfg.model.moe_router_topk == 8
    assert cfg.model.moe_layer_freq == PARAM3_74B_MOE_PATTERN
    assert len(cfg.model.moe_layer_freq) == cfg.model.num_layers

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.checkpoint.load is None
    assert cfg.dataset.blend is None
