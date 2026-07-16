# Param3 74B: dependencies and cluster migration

This document captures the complete environment used for the 32-GPU Param3
74B mock-data run and the optimized performance variants. No model checkpoint,
tokenizer file, or training dataset is required.

## Reproducibility bundle

Carry all three of these artifacts to the destination cluster:

1. This Megatron-Bridge branch: `sniverty/param3-v3-support`.
2. Megatron-Core from NVIDIA/Megatron-LM PR 4624 at commit
   `3b8734896efbf07e542c8eef568de9f8b8c62fce`.
3. The working container image (or a binary-compatible rebuild):
   `nemo_nightly_15072026.sqsh`.

The container used here is:

```text
Size:   39,379,070,976 bytes
SHA256: 51e7796283eacace2a7f1a36451dd98ce976993a2d4e2a0de1babea36dd7437a
```

Verify a copied image before launching:

```bash
echo '51e7796283eacace2a7f1a36451dd98ce976993a2d4e2a0de1babea36dd7437a  /path/to/nemo_nightly_15072026.sqsh' \
  | sha256sum --check
```

Cloning only Megatron-Bridge is insufficient: its Megatron-Core submodule must
be replaced or over-mounted with the PR 4624 tree. The launch script mounts the
selected MCore tree at `/opt/Megatron-Bridge/3rdparty/Megatron-LM`.

## Hardware and scheduler requirements

- Four homogeneous nodes with eight H100 80-GB GPUs per node (32 GPUs total).
- Eight-GPU NVLink domains within each node.
- RDMA/InfiniBand between nodes and a working NCCL fabric.
- Slurm with `sbatch`, `srun`, PMIx, and `gres/gpu` support.
- Pyxis/Enroot support for these `srun` options:
  `--container-image`, `--container-mounts`, `--container-env`,
  `--container-writable`, and `--no-container-mount-home`.
- A shared filesystem visible at the same path on every node for the source,
  sqsh image, NeMo-Run experiment directory, and logs.
- `numactl` inside the container.
- Two CPU NUMA nodes arranged so local GPU ranks 0-3 are local to NUMA node 0
  and ranks 4-7 are local to NUMA node 1. If the new nodes differ, update the
  H100 binding in `scripts/performance/utils/executors.py`.
- A host NVIDIA driver new enough for the CUDA runtime shipped in the image.

The Slurm executor launches one task per GPU and deliberately passes
`--cpu-bind=none`, then prefixes Python with:

```bash
numactl --cpunodebind=$((SLURM_LOCALID/4)) \
        --membind=$((SLURM_LOCALID/4))
```

## Container runtime stack

The known-good sqsh contains the following relevant components:

| Component | Version/revision |
| --- | --- |
| Python | 3.12 |
| PyTorch | `2.13.0a0+8145d630e8.nv26.6.54250401` |
| Triton | `3.7.0+gitb7fa781f.nv26.6` |
| Triton kernels | `1.0.0+gitb7fa781f.nv26.6` |
| Transformer Engine | `2.16.0+d64bc14d` in `/opt/venv` |
| Megatron-Core package metadata | `0.19.0` (overridden by mounted PR source) |
| DeepEP/HybridEP | `1.2.1+17cfb81`, compiled from `17cfb817bccec3a9c247013360cc550c2bac441e` |
| NeMo-Run in the container | `0.11.0+416a0a7` |
| NCCL observed at runtime | `2.30.4+cuda13.2` |

The fused mHC implementation is supplied by the mounted Megatron-Core source,
not by the Megatron-Core wheel. Required source files include:

```text
megatron/core/fusions/fused_mhc_kernels.py
megatron/core/transformer/hyper_connection.py
megatron/core/hyper_comm_grid.py
```

The container also needs FlashQLA/GatedDeltaNet support and is launched with
`FORCE_FLASHQLA_GDN=1`.

The repeated `triton_kernels.matmul_ogs` warning comes from an optional
GPT-OSS inference import in the container's vLLM stack. It is not used by
Param3 training, fused mHC, grouped GEMM, or DeepEP/HybridEP and is not a
required dependency for this run.

## Host-side launcher dependencies

The login node only needs Python 3.12 and NeMo-Run to generate and submit the
Slurm scripts. The known-good launcher uses `nemo-run==0.10.0`:

```bash
cd Megatron-Bridge
python3.12 -m venv .venv-launcher
.venv-launcher/bin/pip install 'nemo-run==0.10.0'
```

Training Python dependencies come from the sqsh, not this small launcher venv.

## Source checkout

```bash
git clone --branch sniverty/param3-v3-support \
  https://github.com/snivertynv/Megatron-Bridge.git

git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-pr4624
git -C Megatron-LM-pr4624 fetch origin \
  pull/4624/head:refs/remotes/origin/pr/4624
git -C Megatron-LM-pr4624 checkout \
  3b8734896efbf07e542c8eef568de9f8b8c62fce
```

Before launching, confirm the mHC file exists:

```bash
test -f Megatron-LM-pr4624/megatron/core/fusions/fused_mhc_kernels.py
```

## Training configuration

Settings common to the verified baseline and performance variants:

```text
Nodes / GPUs:       4 nodes x 8 H100 = 32 GPUs
Sequence length:    4096
Micro batch size:   1
Global batch size:  64
Tensor parallel:    2
Context parallel:   1
Expert parallel:    8
Expert tensor par:  1 (ETP is deliberately disabled)
Data:               mock
Tokenizer:          NullTokenizer, vocab size 128008
Model/grad dtype:   BF16
MoE experts/top-k:  128 / 8
Grouped GEMM:       enabled
Router fusion:      enabled
Permute fusion:     enabled
Fused mHC:          enabled
```

The verified 11.40% MFU baseline is
`param3_74b_pretrain_32gpu_h100_bf16_pp2_precision_aware_config`:

```text
TP=2, PP=2, VP=none, CP=1, EP=8, ETP=1
Dense DP=8, expert DP=2, eight microbatches per step
Selective recompute: layernorm, moe_act, moe, mhc
Dispatcher: alltoall
CUDA graphs: disabled
```

Its warm iterations averaged 4,249.6 ms and 112.78 TFLOP/s/GPU, or 11.40%
MFU against the 989-TFLOP/s H100 BF16 peak.

The optimized recipes remove full routed-MoE recomputation and non-model
benchmark work, then enable:

```text
Selective recompute: layernorm, moe_act, mhc
CUDA graph impl:     transformer_engine
CUDA graph scope:    moe_router, moe_preprocess
Dispatcher:          flex
Flex backend:        hybridep
HybridEP SMs:        32
Precision-aware Adam with BF16 gradients and BF16 moments
```

Available performance variants:

| Variant | Parallelism | Dense/expert DP | Purpose |
| --- | --- | --- | --- |
| `pp2_precision_aware_perf` | TP2, PP2, no VP, EP8, ETP1 | 8 / 2 | Closest optimized baseline |
| `pp2_vp4_precision_aware_perf` | TP2, PP2, VP4, EP8, ETP1 | 8 / 2 | Five layers/chunk; lowest modeled PP2 bubble |
| `pp4_precision_aware_perf` | TP2, PP4, no VP, EP8, ETP1 | 4 / 1 | Lower per-rank model memory |
| `pp4_vp2_precision_aware_perf` | TP2, PP4, VP2, EP8, ETP1 | 4 / 1 | Five layers/chunk with interleaving |

The VP recipes use optimizer-step parameter-gather overlap. They perform no
checkpoint I/O and select `ckpt_format=torch` because MCore does not combine
that overlap mode with distributed checkpointing.

## HybridEP and performance environment

`setup_experiment.py` and its performance plugin set the relevant defaults.
For H100/EP8 the important values are:

```text
NVLINK_DOMAIN_SIZE=8
USE_MNNVL=0
NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
NVTE_FWD_LAYERNORM_SM_MARGIN=20
NVTE_BWD_LAYERNORM_SM_MARGIN=20
CUDA_DEVICE_MAX_CONNECTIONS=1
TORCH_NCCL_AVOID_RECORD_STREAMS=1
TORCH_NCCL_HIGH_PRIORITY=1
NCCL_NVLS_ENABLE=0
NCCL_GRAPH_REGISTER=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORCE_FLASHQLA_GDN=1
NCCL_PXN_DISABLE=1
```

Do not replace the executor launch with a raw `srun` unless these settings and
the NUMA prefix are reproduced. The tracked launcher passes the HybridEP
topology variables explicitly because the performance plugin computes its
environment before CLI recipe overrides are applied.

## Launch on the destination cluster

From the Megatron-Bridge checkout:

```bash
export CONTAINER_IMAGE=/shared/images/nemo_nightly_15072026.sqsh
export MCORE=/shared/src/Megatron-LM-pr4624
export PARAM3_ACCOUNT=<destination-account>
export PARAM3_PARTITION=<destination-partition>
export PARAM3_TIME=00:30:00

# Inspect the generated sbatch and task scripts without submitting.
DRYRUN=1 \
PARAM3_VARIANT=pp2_vp4_precision_aware_perf \
examples/models/param3/launch_performance.sh

# Submit the 32-GPU, GBS64 run.
PARAM3_VARIANT=pp2_vp4_precision_aware_perf \
examples/models/param3/launch_performance.sh
```

The launcher accepts `PARAM3_LOG_DIR`, `LAUNCHER_PYTHON`, `BRIDGE`, and
`PARAM3_EXTRA_MOUNTS` overrides. `PARAM3_EXTRA_MOUNTS` is a comma-separated
Pyxis mount list.

Inspect the generated task script before the first submission and verify it
contains all of the following:

```text
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
--cpu-bind=none
numactl --cpunodebind=$((SLURM_LOCALID/4))
--expert_tensor_parallel_size 1
--global_batch_size 64
--moe_flex_dispatcher_backend hybridep
```

## No external data dependencies

The run uses mock data, random model initialization, and a NullTokenizer.
Leave checkpoint load/save unset for performance measurement. If real data or
a checkpoint is introduced later, it becomes an additional shared-filesystem
dependency and checkpoint compatibility must be checked against the exact
Megatron-Core revision above.
