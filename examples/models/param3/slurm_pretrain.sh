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

# Param3 74B mock-data pretraining. The container must include the Param3
# Megatron-Core branch providing mHC and FlashQLA GatedDeltaNet support.

#SBATCH --job-name=param3-74b
#SBATCH --account=sw_aidot
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --output=param3_74b_%j.out
#SBATCH --error=param3_74b_%j.err
#SBATCH --exclusive

set -euo pipefail

MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-/opt/Megatron-Bridge}
WORKSPACE=${WORKSPACE:-/workspace}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${WORKSPACE}/results/param3_74b_${SLURM_JOB_ID}/checkpoints}
RECIPE_NAME=${RECIPE_NAME:-param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config}

if [ -z "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: CONTAINER_IMAGE must point to the Param3 mHC container."
    exit 1
fi

export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export FORCE_FLASHQLA_GDN=${FORCE_FLASHQLA_GDN:-1}
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# H100 NVL8 settings used by the Megatron performance launcher for HybridEP.
# They are inert for recipes that retain the all-to-all dispatcher.
export NVLINK_DOMAIN_SIZE=${NVLINK_DOMAIN_SIZE:-8}
export USE_MNNVL=${USE_MNNVL:-0}
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-128}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-20}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-20}
export MEGATRON_BRIDGE_PATH WORKSPACE CHECKPOINT_DIR RECIPE_NAME

if [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    MASTER_ADDR=${MASTER_ADDR%%$'\n'*}
fi
if [ -z "${MASTER_ADDR}" ]; then
    echo "ERROR: failed to resolve MASTER_ADDR from ${SLURM_JOB_NODELIST}." >&2
    exit 1
fi
export MASTER_ADDR

read -r -d '' INNER_SCRIPT <<'EOF' || true
set -euo pipefail
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}

cd "${MEGATRON_BRIDGE_PATH}"
numactl --cpunodebind=$((SLURM_LOCALID / 4)) --membind=$((SLURM_LOCALID / 4)) \
    uv run --no-sync python scripts/training/run_recipe.py \
    --recipe "${RECIPE_NAME}" \
    --dataset llm-pretrain-mock \
    --step_func gpt_step \
    checkpoint.save="${CHECKPOINT_DIR}" \
    checkpoint.load=null
EOF

# Let numactl own CPU affinity.  task/affinity may otherwise pin each task
# before numactl can bind ranks 0-3 and 4-7 to their respective NUMA nodes.
SRUN_CMD=(srun --mpi=pmix --cpu-bind=none --kill-on-bad-exit=1 --container-image="${CONTAINER_IMAGE}")
if [ -n "${CONTAINER_MOUNTS}" ]; then
    SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

echo "Launching Param3 74B on ${SLURM_NNODES} nodes / ${SLURM_NTASKS} GPUs"
echo "Recipe: ${RECIPE_NAME}"
echo "Run uses mock data, global batch size 64, and random initialization"
echo "Distributed rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
"${SRUN_CMD[@]}" bash -lc "${INNER_SCRIPT}"
