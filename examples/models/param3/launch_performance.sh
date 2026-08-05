#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Launch a Param3 performance recipe through setup_experiment.py.  The Slurm
# executor supplies per-rank H100 NUMA binding and the H100 HybridEP settings.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BRIDGE=${BRIDGE:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
MCORE=${MCORE:-${BRIDGE}/3rdparty/Megatron-LM}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
LAUNCHER_PYTHON=${LAUNCHER_PYTHON:-${BRIDGE}/.venv-launcher/bin/python}

PARAM3_VARIANT=${PARAM3_VARIANT:-pp2_precision_aware_perf}
PARAM3_ACCOUNT=${PARAM3_ACCOUNT:-nemotron_n4_pre}
PARAM3_PARTITION=${PARAM3_PARTITION:-backfill}
PARAM3_TIME=${PARAM3_TIME:-00:30:00}
PARAM3_LOG_DIR=${PARAM3_LOG_DIR:-${BRIDGE}/nemo_runs/${PARAM3_VARIANT}}
PARAM3_EXTRA_MOUNTS=${PARAM3_EXTRA_MOUNTS:-}
PARAM3_COMPILE_CACHE=${PARAM3_COMPILE_CACHE:-}
PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1=${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1:-0}
PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2=${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2:-0}
PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_BWD=${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_BWD:-0}
PARAM3_ENABLE_SM90_CUTEDSL_DIRECT_LAYOUTS=${PARAM3_ENABLE_SM90_CUTEDSL_DIRECT_LAYOUTS:-0}
PARAM3_ENABLE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION=${PARAM3_ENABLE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION:-0}
PARAM3_SM90_CUTEDSL_DIR=${PARAM3_SM90_CUTEDSL_DIR:-${SCRIPT_DIR}/cutedsl_sm90}
PARAM3_SM90_CUTEDSL_FC2_FORWARD_TILE=${PARAM3_SM90_CUTEDSL_FC2_FORWARD_TILE:-}
PARAM3_SM90_CUTEDSL_FC2_FORWARD_CLUSTER=${PARAM3_SM90_CUTEDSL_FC2_FORWARD_CLUSTER:-}
PARAM3_SM90_CUTEDSL_FC2_FORWARD_RASTER=${PARAM3_SM90_CUTEDSL_FC2_FORWARD_RASTER:-}
PARAM3_SM90_CUTEDSL_FC2_DGRAD_TILE=${PARAM3_SM90_CUTEDSL_FC2_DGRAD_TILE:-}
PARAM3_SM90_CUTEDSL_FC2_DGRAD_CLUSTER=${PARAM3_SM90_CUTEDSL_FC2_DGRAD_CLUSTER:-}
PARAM3_SM90_CUTEDSL_FC2_DGRAD_RASTER=${PARAM3_SM90_CUTEDSL_FC2_DGRAD_RASTER:-}
PARAM3_SM90_CUTEDSL_FC2_WGRAD_TILE=${PARAM3_SM90_CUTEDSL_FC2_WGRAD_TILE:-}
PARAM3_SM90_CUTEDSL_FC2_WGRAD_CLUSTER=${PARAM3_SM90_CUTEDSL_FC2_WGRAD_CLUSTER:-}
PARAM3_SM90_CUTEDSL_FC2_WGRAD_RASTER=${PARAM3_SM90_CUTEDSL_FC2_WGRAD_RASTER:-}
PARAM3_SM90_CUTEDSL_FC2_WGRAD_ACCUMULATE_TILE=${PARAM3_SM90_CUTEDSL_FC2_WGRAD_ACCUMULATE_TILE:-}

case "${PARAM3_VARIANT}" in
    pp2_*) PP_SIZE=2 ;;
    pp4_*) PP_SIZE=4 ;;
    *)
        echo "ERROR: unsupported Param3 variant: ${PARAM3_VARIANT}" >&2
        exit 1
        ;;
esac

if [[ -z "${CONTAINER_IMAGE}" ]]; then
    echo "ERROR: set CONTAINER_IMAGE to the Param3/HybridEP sqsh image." >&2
    exit 1
fi

for path in "${BRIDGE}" "${MCORE}" "${CONTAINER_IMAGE}" "${LAUNCHER_PYTHON}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ ! -f "${MCORE}/megatron/core/fusions/fused_mhc_kernels.py" ]]; then
    echo "ERROR: MCORE does not contain the fused mHC implementation from NVIDIA/Megatron-LM PR 4624." >&2
    exit 1
fi

mkdir -p "${PARAM3_LOG_DIR}"

MOUNTS="${BRIDGE}:/opt/Megatron-Bridge,${MCORE}:/opt/Megatron-Bridge/3rdparty/Megatron-LM"
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1}" == 1 || "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2}" == 1 ]]; then
    if [[ ! -f "${PARAM3_SM90_CUTEDSL_DIR}/param3_sm90_grouped_linear.py" ]]; then
        echo "ERROR: PARAM3_SM90_CUTEDSL_DIR must contain param3_sm90_grouped_linear.py." >&2
        exit 1
    fi
    MOUNTS+=",${PARAM3_SM90_CUTEDSL_DIR}:/opt/param3-cutedsl"
fi
if [[ -n "${PARAM3_EXTRA_MOUNTS}" ]]; then
    MOUNTS+=",${PARAM3_EXTRA_MOUNTS}"
fi

if [[ -n "${PARAM3_COMPILE_CACHE}" ]]; then
    mkdir -p "${PARAM3_COMPILE_CACHE}"
    MOUNTS+=",${PARAM3_COMPILE_CACHE}:${PARAM3_COMPILE_CACHE}"
fi

ARGS=(
    "${BRIDGE}/scripts/performance/setup_experiment.py"
    --account "${PARAM3_ACCOUNT}"
    --partition "${PARAM3_PARTITION}"
    --time_limit "${PARAM3_TIME}"
    --container_image "${CONTAINER_IMAGE}"
    --model_family_name param3
    --model_recipe_name param3_74b
    --config_variant "${PARAM3_VARIANT}"
    --task pretrain
    --gpu h100
    --compute_dtype bf16
    --num_gpus 32
    --gpus_per_node 8
    --gres gpu:8
    --tensor_model_parallel_size 2
    --pipeline_model_parallel_size "${PP_SIZE}"
    --context_parallel_size 1
    --expert_model_parallel_size 8
    --expert_tensor_parallel_size 1
    --global_batch_size 64
    --micro_batch_size 1
    --seq_length 4096
    --data mock
    --tokenizer_type NullTokenizer
    --vocab_size 128008
    --cuda_graph_impl transformer_engine
    --cuda_graph_scope attn,moe_router,moe_preprocess
    --moe_flex_dispatcher_backend hybridep
    --wandb_experiment_name "param3_74b_${PARAM3_VARIANT}"
    --log_dir "${PARAM3_LOG_DIR}"
    --custom_mounts "${MOUNTS}"
    --custom_srun_args=--cpu-bind=none
    -E FORCE_FLASHQLA_GDN=1
    -E NCCL_PXN_DISABLE=1
    -E CUDA_DEVICE_MAX_CONNECTIONS=1
    -E NVLINK_DOMAIN_SIZE=8
    -E USE_MNNVL=0
    -E NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
    -E NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
    -E NVTE_FWD_LAYERNORM_SM_MARGIN=20
    -E NVTE_BWD_LAYERNORM_SM_MARGIN=20
    --offline
    --dump_env
    --max_retries 0
    --packager none
    --detach true
)

# setup_experiment.py only changes recompute when this argument is present.
# Leave it absent for the PP4/VP2 memory-headroom probe so the recipe's
# recompute_granularity=None and recompute_modules=[] survive CLI overrides.
if [[ "${PARAM3_VARIANT}" != *no_recompute* ]]; then
    ARGS+=(--recompute_modules layernorm,moe_act,mhc)
fi

if [[ -n "${PARAM3_COMPILE_CACHE}" ]]; then
    ARGS+=(-E "PERF_COMPILE_CACHE_ROOT=${PARAM3_COMPILE_CACHE}")
fi

if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1}" == 1 || "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2}" == 1 ]]; then
    if [[ -z "${PARAM3_COMPILE_CACHE}" ]]; then
        echo "ERROR: SM90 CuTeDSL grouped experts require PARAM3_COMPILE_CACHE." >&2
        exit 1
    fi
    CUTEDSL_CACHE_DIR="${PARAM3_COMPILE_CACHE}/cutedsl"
    mkdir -p "${CUTEDSL_CACHE_DIR}"
    ARGS+=(
        -E MCORE_SM90_CUTEDSL_ADAPTER_DIR=/opt/param3-cutedsl
        -E MCORE_SM90_CUTEDSL_MAX_TOKENS=32768
        -E CUTE_DSL_NO_CACHE=0
        -E "CUTE_DSL_CACHE_DIR=${CUTEDSL_CACHE_DIR}"
    )
fi
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1}" == 1 ]]; then
    ARGS+=(-E MCORE_SM90_CUTEDSL_GROUPED_FC1=1)
fi
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2}" == 1 ]]; then
    ARGS+=(-E MCORE_SM90_CUTEDSL_GROUPED_FC2=1)
    for tuning_name in \
        PARAM3_SM90_CUTEDSL_FC2_FORWARD_TILE \
        PARAM3_SM90_CUTEDSL_FC2_FORWARD_CLUSTER \
        PARAM3_SM90_CUTEDSL_FC2_FORWARD_RASTER \
        PARAM3_SM90_CUTEDSL_FC2_DGRAD_TILE \
        PARAM3_SM90_CUTEDSL_FC2_DGRAD_CLUSTER \
        PARAM3_SM90_CUTEDSL_FC2_DGRAD_RASTER \
        PARAM3_SM90_CUTEDSL_FC2_WGRAD_TILE \
        PARAM3_SM90_CUTEDSL_FC2_WGRAD_CLUSTER \
        PARAM3_SM90_CUTEDSL_FC2_WGRAD_RASTER \
        PARAM3_SM90_CUTEDSL_FC2_WGRAD_ACCUMULATE_TILE
    do
        if [[ -n "${!tuning_name}" ]]; then
            ARGS+=(-E "MCORE_${tuning_name#PARAM3_}=${!tuning_name}")
        fi
    done
fi
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_BWD}" == 1 ]]; then
    if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1}" != 1 && "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2}" != 1 ]]; then
        echo "ERROR: grouped backward requires FC1 or FC2 grouped GEMM." >&2
        exit 1
    fi
    ARGS+=(-E MCORE_SM90_CUTEDSL_GROUPED_BWD=1)
fi
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_DIRECT_LAYOUTS}" == 1 ]]; then
    if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_BWD}" != 1 ]]; then
        echo "ERROR: direct layouts require grouped backward." >&2
        exit 1
    fi
    ARGS+=(-E MCORE_SM90_CUTEDSL_DIRECT_LAYOUTS=1)
fi
if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION}" == 1 ]]; then
    if [[ "${PARAM3_ENABLE_SM90_CUTEDSL_DIRECT_LAYOUTS}" != 1 ]]; then
        echo "ERROR: fused wgrad accumulation requires direct layouts." >&2
        exit 1
    fi
    ARGS+=(-E MCORE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION=1)
fi

if [[ "${DRYRUN:-0}" == 1 ]]; then
    ARGS+=(--dryrun)
fi

export PYTHONPATH="${BRIDGE}/src:${BRIDGE}/scripts/performance${PYTHONPATH:+:${PYTHONPATH}}"
exec "${LAUNCHER_PYTHON}" "${ARGS[@]}"
