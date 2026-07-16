# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import torch

from megatron.bridge.perf_recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config as flat_pp2_perf_config,
)
from megatron.bridge.perf_recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config as flat_pp2_vp4_perf_config,
)
from megatron.bridge.perf_recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config as flat_pp4_perf_config,
)
from megatron.bridge.perf_recipes.param3 import (
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config as flat_pp4_vp2_perf_config,
)
from megatron.bridge.recipes.param3 import (
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
    param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config,
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
    assert cfg.model.use_fused_mhc is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.model.recompute_modules == ["layernorm", "moe_act", "moe", "mhc"]

    assert cfg.model.num_moe_experts == 128
    assert cfg.model.moe_router_topk == 8
    assert cfg.model.moe_layer_freq == PARAM3_74B_MOE_PATTERN
    assert len(cfg.model.moe_layer_freq) == cfg.model.num_layers

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.train_iters is None
    assert cfg.train.train_samples == 1600
    assert cfg.train.global_batch_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.checkpoint.load is None
    assert cfg.dataset.blend is None


def test_param3_74b_cutedsl_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config()

    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_modules == []
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_shared_expert_overlap is True


def test_param3_74b_fsdp_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_fsdp_config()

    assert cfg.dist.use_megatron_fsdp is True
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.average_in_collective is False
    assert cfg.model.init_model_with_meta_device is False
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"


def test_param3_74b_pp2_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1


def test_param3_74b_pp2_no_moe_recompute_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_no_moe_recompute_config()

    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.recompute_modules == ["layernorm", "moe_act", "mhc"]


def test_param3_74b_pp2_precision_aware_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config()

    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.optimizer.bf16 is True
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.store_param_remainders is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_param3_74b_pp2_precision_aware_perf_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config()

    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.recompute_modules == ["layernorm", "moe_act", "mhc"]
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.checkpoint.save is None
    assert cfg.validation.eval_iters == 0
    assert cfg.logger.log_params_norm is False
    assert cfg.logger.barrier_with_L1_time is False
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.rerun_state_machine.check_for_nan_in_loss is False
    assert cfg.optimizer.barrier_with_L1_time is False
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["moe_router", "moe_preprocess"]
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_a2a_overlap is False
    assert cfg.model.moe_shared_expert_overlap is False


def test_param3_74b_pp4_precision_aware_perf_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.train.global_batch_size == 64
    assert cfg.model.recompute_modules == ["layernorm", "moe_act", "mhc"]


def test_param3_flat_perf_recipe_exports() -> None:
    pp2_cfg = flat_pp2_perf_config()
    pp4_cfg = flat_pp4_perf_config()

    assert pp2_cfg.model.pipeline_model_parallel_size == 2
    assert pp4_cfg.model.pipeline_model_parallel_size == 4
    assert pp2_cfg.model.expert_tensor_parallel_size == 1
    assert pp4_cfg.model.expert_tensor_parallel_size == 1


def test_param3_74b_pp4_vp2_precision_aware_perf_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config()
    flat_cfg = flat_pp4_vp2_perf_config()

    for recipe in (cfg, flat_cfg):
        assert recipe.model.pipeline_model_parallel_size == 4
        assert recipe.model.virtual_pipeline_model_parallel_size == 2
        assert recipe.model.expert_tensor_parallel_size == 1
        assert recipe.checkpoint.ckpt_format == "torch"
        assert recipe.optimizer.overlap_param_gather_with_optimizer_step is True
        assert recipe.comm_overlap.tp_comm_overlap is False
        assert recipe.comm_overlap.overlap_param_gather_with_optimizer_step is True


def test_param3_74b_pp2_vp4_precision_aware_perf_recipe() -> None:
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config()
    flat_cfg = flat_pp2_vp4_perf_config()

    for recipe in (cfg, flat_cfg):
        assert recipe.model.pipeline_model_parallel_size == 2
        assert recipe.model.virtual_pipeline_model_parallel_size == 4
        assert recipe.model.expert_tensor_parallel_size == 1
        assert recipe.train.train_samples == 768
        assert recipe.checkpoint.ckpt_format == "torch"
        assert recipe.optimizer.overlap_param_gather_with_optimizer_step is True
        assert recipe.comm_overlap.tp_comm_overlap is False
        assert recipe.comm_overlap.overlap_param_gather_with_optimizer_step is True
