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

"""Param3 pretraining recipes."""

import torch
import torch.nn.functional as F

from megatron.bridge.models.param3 import Param3ModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing_samples
from megatron.bridge.training.config import ConfigContainer


PARAM3_74B_ATTENTION_PATTERN = [0, 1, 1] * 13 + [0]
PARAM3_74B_MOE_PATTERN = [0] + [1] * 39


def param3_74b_pretrain_32gpu_h100_bf16_config() -> ConfigContainer:
    """Return the logged 32-GPU BF16 pretraining configuration for Param3 74B.

    The recipe uses TP=2, PP=1, EP=8 and mock data. Attention pattern values
    use Megatron-Core semantics: 0 is full attention and 1 is GatedDeltaNet.
    It requires a Megatron-Core revision containing mHC recomputation support.

    Returns:
        ConfigContainer configured for the Param3 74B pretraining smoke run.
    """
    cfg = _pretrain_common()

    cfg.model = Param3ModelProvider(
        num_layers=40,
        hidden_size=3072,
        ffn_hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=128008,
        seq_length=4096,
        max_position_embeddings=4096,
        make_vocab_size_divisible_by=1,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        position_embedding_type="rope",
        apply_rope_fusion=True,
        attention_output_gate=True,
        attention_softmax_in_fp32=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.01,
        embedding_init_method_std=0.01,
        autocast_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp16=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=PARAM3_74B_ATTENTION_PATTERN.copy(),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=32,
        linear_num_value_heads=64,
        num_moe_experts=128,
        moe_layer_freq=PARAM3_74B_MOE_PATTERN.copy(),
        moe_ffn_hidden_size=1536,
        moe_router_topk=8,
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-3,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_shared_expert_intermediate_size=1536,
        moe_shared_expert_overlap=True,
        mtp_loss_scaling_factor=0.0,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        microbatch_group_size_per_vp_stage=1,
        hetereogenous_dist_checkpoint=True,
        gradient_accumulation_fusion=True,
        cross_entropy_loss_fusion=True,
        masked_softmax_fusion=True,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
        kitchen_attention_backend=None,
    )

    # ``mhc`` is supplied by the Param3 Megatron-Core branch. Assign it after
    # provider construction so this recipe remains inspectable with upstream
    # Megatron-Core revisions that have not yet added that recompute module.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["layernorm", "moe_act", "moe", "mhc"]

    # Mock data and tokenizer: no dataset or checkpoint download is required.
    cfg.dataset.blend = None
    cfg.dataset.split = "99,1,0"
    cfg.dataset.random_seed = 42
    cfg.dataset.seq_length = 4096
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = 128008

    cfg.train.train_iters = 25
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 50
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 20

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing_samples(
        lr_warmup_samples=1599,
        lr_decay_samples=1600,
        max_lr=5e-5,
        min_lr=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads"
    cfg.ddp.pad_buckets_for_high_nccl_busbw = True
    cfg.ddp.use_distributed_optimizer = True

    # Start from random initialization. New checkpoints may still be saved.
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 30000
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    cfg.logger.log_interval = 1
    cfg.logger.log_params_norm = True
    cfg.logger.log_throughput = True
    cfg.rng.seed = 1234

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config() -> ConfigContainer:
    """Return the Param3 recipe with cuTeDSL routed-expert MoE fusion.

    This performance variant keeps the logged Param3 architecture and training
    settings while enabling the Transformer Engine op fuser and its required
    GLU interleaving. The cuTeDSL fusion applies to routed experts only; CUDA
    graphs remain disabled.

    Returns:
        ConfigContainer configured for the optimized Param3 74B smoke run.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_config()

    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    return cfg


__all__ = [
    "PARAM3_74B_ATTENTION_PATTERN",
    "PARAM3_74B_MOE_PATTERN",
    "param3_74b_pretrain_32gpu_h100_bf16_config",
    "param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config",
]
