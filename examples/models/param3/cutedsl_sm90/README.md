# Param3 SM90 BF16 CuTeDSL grouped experts

This directory contains the direct, eager-execution SM90 CuTeDSL grouped-GEMM
implementation validated for Param3 on H100. It combines all local expert
problems into one persistent kernel launch for each of FC1 forward, FC1 dgrad,
FC1 wgrad, FC2 forward, FC2 dgrad, and FC2 wgrad. SwiGLU remains a separate
operation. The optional wgrad epilogue accumulates directly into BF16
`main_grad`.

This package intentionally excludes:

- Transformer Engine GroupedTensor backend replacement
- device-resident metadata generation
- full-iteration CUDA graph support
- CUDA graph metadata-slot rings and graph validation
- MoE paged stash
- token dropping, forced load balancing, or expert-capacity padding

## Pinned source versions

- Megatron-Bridge base: `13a1be0b91a3510a722c5743019ea4261a4fb7af`
- Megatron-Core base: `3b8734896efbf07e542c8eef568de9f8b8c62fce`
- CUTLASS base: `e406c186f510a15091cce01f782020ceb7ba8eb5`
  (`v4.5.0`)

The runtime loads the bundled modified CUTLASS example from:

```text
cutlass-v4.5.0/examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py
```

The container must provide the matching CUTLASS/CuTeDSL Python package. The
modified example is bundled locally, so the complete CUTLASS checkout is not
required at runtime.

## Apply the Megatron-Core integration

Start from the pinned Megatron-Core commit and apply:

```bash
git -C /path/to/Megatron-LM checkout 3b8734896efbf07e542c8eef568de9f8b8c62fce
git -C /path/to/Megatron-LM apply \
  /path/to/Megatron-Bridge/examples/models/param3/cutedsl_sm90/patches/megatron-core-sm90-cutedsl-experts.patch
```

The CUTLASS patch is included for review or for maintaining a complete matching
CUTLASS checkout:

```bash
git -C /path/to/cutlass-v4.5.0 checkout e406c186f510a15091cce01f782020ceb7ba8eb5
git -C /path/to/cutlass-v4.5.0 apply \
  /path/to/Megatron-Bridge/examples/models/param3/cutedsl_sm90/patches/cutlass-v4.5.0-sm90-bf16-grouped-gemm.patch
```

Do not both apply the CUTLASS patch and replace the patched file with the
bundled file; they represent the same source change.

## Launch

Set the cluster-specific values plus:

```bash
export MCORE=/path/to/patched/Megatron-LM
export PARAM3_COMPILE_CACHE=/shared/writable/cutedsl-cache
export PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC1=1
export PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_FC2=1
export PARAM3_ENABLE_SM90_CUTEDSL_GROUPED_BWD=1
export PARAM3_ENABLE_SM90_CUTEDSL_DIRECT_LAYOUTS=1
export PARAM3_ENABLE_SM90_CUTEDSL_FUSED_WGRAD_ACCUMULATION=1
examples/models/param3/launch_performance.sh
```

`launch_performance.sh` mounts this directory at `/opt/param3-cutedsl`, exports
the MCore feature switches, and enables a reusable CuTeDSL JIT cache. The
selected recipe must keep `model.use_transformer_engine_op_fuser=false`; the
bundled default `pp2_precision_aware_perf` recipe already does.

The validated default tuning is:

| Phase | Tile | Cluster | Raster |
| --- | --- | --- | --- |
| Forward | 128x256 | 2x1 | M |
| Dgrad | 128x256 | 2x1 | M |
| Wgrad overwrite | 128x256 | 2x1 | N |
| Wgrad accumulate | 128x256 | 1x1 | M |

FC2 tuning can be overridden using the `PARAM3_SM90_CUTEDSL_FC2_*`
environment variables defined in `launch_performance.sh`.

## Validation

Syntax checks do not require a GPU:

```bash
bash -n examples/models/param3/launch_performance.sh
python -m py_compile \
  examples/models/param3/cutedsl_sm90/param3_sm90_grouped_linear.py \
  examples/models/param3/cutedsl_sm90/param3_sm90_grouped_backward.py \
  examples/models/param3/cutedsl_sm90/param3_packed_transpose.py
```

Numerical validation must run inside the matching CUDA container on an H100:

```bash
cd examples/models/param3/cutedsl_sm90
PYTHONPATH=. python validation/validate_sm90_grouped_backward.py
PYTHONPATH=. python validation/validate_sm90_grouped_linear_autograd.py
```

The full 32-GPU training validation corresponding to this direct path reached
2.502850 seconds per step, 191.417 TFLOP/s/GPU, and 19.3546% MFU.
