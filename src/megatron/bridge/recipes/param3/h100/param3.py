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
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


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
        use_fused_mhc=True,
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

    cfg.train.train_iters = None
    cfg.train.train_samples = 1600
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

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    # The local Param3 shard is large enough that an FP32 gradient buffer alone
    # consumes ~33 GiB and OOMs an 80-GiB H100 during DDP construction.  Reduce
    # gradients in BF16 so the buffer stays at the model precision (~16.5 GiB).
    cfg.ddp.grad_reduce_in_fp32 = False
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


def param3_74b_pretrain_32gpu_h100_bf16_fsdp_config() -> ConfigContainer:
    """Return the BF16 Param3 recipe with fully sharded model state.

    This debug/performance variant keeps the unfused H100 expert path and uses
    Megatron FSDP to reduce the model-state footprint.  It is intentionally
    separate from the parity recipe so queued DDP runs remain unchanged.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_config()

    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.average_in_collective = False
    # Param3's TransformerBlock does not expose a reset method, so Megatron
    # FSDP cannot materialize it from meta tensors on this MCore revision.
    cfg.model.init_model_with_meta_device = False
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp2_config() -> ConfigContainer:
    """Return the BF16 Param3 recipe split across two pipeline stages."""
    cfg = param3_74b_pretrain_32gpu_h100_bf16_config()

    cfg.model.pipeline_model_parallel_size = 2

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config() -> ConfigContainer:
    """Return the PP2 recipe with reduced-memory TE FusedAdam state.

    Precision-aware Adam stores BF16 master-weight remainders and BF16 first
    and second moments.  This reduces the persistent distributed-optimizer
    footprint without changing the model-parallel topology.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_config()

    cfg.optimizer.bf16 = True
    cfg.optimizer.params_dtype = torch.bfloat16
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.store_param_remainders = True
    cfg.optimizer.main_params_dtype = torch.float32
    # Gradients are already reduced in BF16 by this recipe.  Avoid materializing
    # an additional FP32 main-gradient shard solely for the optimizer step.
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    return cfg


def _apply_param3_perf_measurement_overrides(cfg: ConfigContainer) -> None:
    """Remove training-loop work that is not part of the model step.

    Keep per-iteration throughput logging so short Slurm runs remain directly
    measurable, but avoid parameter-norm reductions, NaN scans, timing
    barriers, validation, TensorBoard writes, and checkpoint serialization.
    """
    cfg.checkpoint.save = None
    cfg.validation.eval_iters = 0
    cfg.train.eval_iters = 0

    cfg.logger.tensorboard_dir = None
    cfg.logger.log_params_norm = False
    cfg.logger.log_timers_to_tensorboard = False
    cfg.logger.barrier_with_L1_time = False

    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.check_for_large_grads = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.optimizer.barrier_with_L1_time = False

    # Match the proven Qwen3-MoE H100 performance path: capture the dynamic
    # router/preprocess launch sequence and use HybridEP's flex dispatcher.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_a2a_overlap = False
    cfg.model.moe_shared_expert_overlap = False
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True


def param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config() -> ConfigContainer:
    """Return the PP2 precision-aware throughput configuration.

    This keeps TP=2, PP=2, EP=8, and ETP=1 from the passing 32-GPU baseline,
    while removing the full routed-MoE recomputation and benchmark-only work.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config()

    cfg.model.recompute_modules = ["layernorm", "moe_act", "mhc"]
    _apply_param3_perf_measurement_overrides(cfg)

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config() -> ConfigContainer:
    """Return the interleaved PP2 throughput configuration.

    Four virtual chunks put five layers in each chunk and reduce the ideal
    PP2 pipeline bubble from 1/9 to 1/33 for the eight microbatches used at
    global batch size 64.  Interleaving also enables optimizer-step parameter
    gather overlap while preserving EP=8 and ETP=1.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config()

    cfg.model.virtual_pipeline_model_parallel_size = 4
    # The first iteration performs the expensive kernel/graph warmup.  Twelve
    # iterations leave ten steady-state samples while fitting a 10-minute
    # backfill probe; promising results are rerun for the full 25 iterations.
    cfg.train.train_samples = 12 * cfg.train.global_batch_size
    # MCore currently disallows optimizer-step parameter-gather overlap when
    # use_dist_ckpt is true.  This measurement recipe neither loads nor saves a
    # checkpoint, so select the legacy format solely to clear that runtime flag.
    cfg.checkpoint.ckpt_format = "torch"
    cfg.optimizer.overlap_param_gather_with_optimizer_step = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_param_gather_with_optimizer_step=True,
    )

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config() -> ConfigContainer:
    """Return a lower-memory PP4 variant of the throughput configuration.

    EP remains eight, so PP4 reduces the local routed-expert parameter and
    gradient footprint while preserving ETP=1.  On 32 GPUs this topology uses
    dense DP=4, expert DP=1, and 16 microbatches for global batch size 64.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config()

    cfg.model.pipeline_model_parallel_size = 4

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config() -> ConfigContainer:
    """Return the interleaved PP4 throughput configuration.

    Two virtual chunks put five layers in each chunk and reduce the pipeline
    bubble for the 16 microbatches used at global batch size 64.  The virtual
    chunks also make optimizer-step parameter-gather overlap applicable.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config()

    cfg.model.virtual_pipeline_model_parallel_size = 2
    # See the PP2/VP4 variant above: no checkpoint I/O is performed, and this
    # format keeps optimizer-step parameter-gather overlap valid in MCore.
    cfg.checkpoint.ckpt_format = "torch"
    cfg.optimizer.overlap_param_gather_with_optimizer_step = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_param_gather_with_optimizer_step=True,
    )

    return cfg


def param3_74b_pretrain_32gpu_h100_bf16_pp2_no_moe_recompute_config() -> ConfigContainer:
    """Return the PP2 recipe without full routed-MoE recomputation.

    PP2 leaves enough activation headroom on 80-GiB H100s to retain only the
    fine-grained layernorm, MoE activation, and mHC recomputations.  Avoiding a
    second full MoE forward in backward is expected to improve throughput while
    keeping expert tensor parallelism disabled.
    """
    cfg = param3_74b_pretrain_32gpu_h100_bf16_pp2_config()

    cfg.model.recompute_modules = ["layernorm", "moe_act", "mhc"]

    return cfg


__all__ = [
    "PARAM3_74B_ATTENTION_PATTERN",
    "PARAM3_74B_MOE_PATTERN",
    "param3_74b_pretrain_32gpu_h100_bf16_config",
    "param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config",
    "param3_74b_pretrain_32gpu_h100_bf16_fsdp_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_vp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp4_vp2_precision_aware_perf_config",
    "param3_74b_pretrain_32gpu_h100_bf16_pp2_no_moe_recompute_config",
]
