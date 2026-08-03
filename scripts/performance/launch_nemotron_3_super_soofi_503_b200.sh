#!/bin/bash
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

set -euo pipefail

: "${SOOFI_ACCOUNT:?Set SOOFI_ACCOUNT to the Slurm account}"
: "${SOOFI_PARTITION:?Set SOOFI_PARTITION to the Slurm partition}"
: "${SOOFI_CONTAINER_IMAGE:?Set SOOFI_CONTAINER_IMAGE to the validated NeMo container}"

SOOFI_LOG_DIR="${SOOFI_LOG_DIR:-${PWD}/results/nemotron_3_super_soofi_503}"
SOOFI_TIME_LIMIT="${SOOFI_TIME_LIMIT:-00:30:00}"

uv run python scripts/performance/setup_experiment.py \
  --account "${SOOFI_ACCOUNT}" \
  --partition "${SOOFI_PARTITION}" \
  --container_image "${SOOFI_CONTAINER_IMAGE}" \
  --model_family_name nemotronh \
  --model_recipe_name nemotron_3_super \
  --task pretrain \
  --gpu b200 \
  --compute_dtype nvfp4 \
  --num_gpus 64 \
  --gpus_per_node 8 \
  --time_limit "${SOOFI_TIME_LIMIT}" \
  --log_dir "${SOOFI_LOG_DIR}" \
  --custom_env_vars \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_GRAPH_REGISTER=0,NCCL_NVLS_ENABLE=0,TORCH_NCCL_AVOID_RECORD_STREAMS=1,TORCH_NCCL_HIGH_PRIORITY=1,CUDA_DEVICE_MAX_CONNECTIONS=32" \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=1 \
  model.expert_tensor_parallel_size=1 \
  model.expert_model_parallel_size=32 \
  train.micro_batch_size=1 \
  train.global_batch_size=512 \
  model.recompute_modules=[] \
  model.recompute_granularity=selective \
  model.moe_router_force_load_balancing=false \
  model.cross_entropy_fusion_impl=native \
  model.use_transformer_engine_op_fuser=false \
  mixed_precision.fp4_param_gather=false \
  ddp.fp4_param_gather=false \
  mixed_precision.grad_reduce_in_fp32=true \
  ddp.grad_reduce_in_fp32=true \
  ddp.bucket_size=134217728 \
  optimizer.use_precision_aware_optimizer=true \
  optimizer.store_param_remainders=false \
  checkpoint.async_save=false \
  ddp.check_for_nan_in_grad=false \
  ddp.check_for_large_grads=false \
  "$@"
