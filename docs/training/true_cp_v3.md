# Mamba True Context Parallel v3

This branch provides an opt-in Mamba True Context Parallel (True CP) training
path for Nemotron-3 Super. It was validated with the NeMo 26.06 container on
128 B200 GPUs using TP2 / CP64 / PP1 / EP32 and a 1,048,576-token sequence.

True CP leaves each Mamba sequence sharded across the context-parallel ranks.
It replaces the native CP-to-head exchanges with one compact boundary
all-gather in forward and one in backward. The attention, MoE, layer ordering,
and standard Mamba model spec remain unchanged.

## Checkout

Clone this branch and initialize the pinned Megatron-LM submodule:

```bash
git clone --branch sniverty/mamba-true-cp-v3 \
  --recurse-submodules https://github.com/snivertynv/Megatron-Bridge.git
```

If the repository was cloned without `--recurse-submodules`, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The submodule is pinned to commit
`0f70e8cfffa03acaccec07121131ecb991747bf0` on
`snivertynv/Megatron-LM:sniverty/mamba-true-cp-v3`.

## Enable True CP

Set these environment variables in every training process before constructing
the Nemotron-3 Super performance recipe:

```bash
export SOOFI_ENABLE_MAMBA_TRUE_CP_V3=1
export SOOFI_MAMBA_TRUE_CP_CHUNK_SIZE=256
export SOOFI_MAMBA_TRUE_CP_BWD_SUBSEQUENCE_SIZE=64
```

The validated topology was:

```text
tensor model parallel size:  2
pipeline model parallel size: 1
context parallel size:       64
expert model parallel size:  32
micro batch size:             1
global batch size:           16
sequence length:       1,048,576
```

`PP1` means no pipeline partitioning. True CP changes only the Mamba mixer; it
does not require pipeline parallelism.

## Current scope

The implementation currently supports training with:

- context parallel size greater than one;
- width-4 Mamba convolution;
- SiLU activation;
- `D_has_hdim=False`; and
- the memory-efficient Mamba path.

Inference and packed sequences are not yet supported. The implementation is
kept behind an environment-variable opt-in, so ordinary Nemotron-3 Super
recipes continue to use the standard Mamba path.

Optional `record_function` ranges are disabled by default. Do not enable them
for performance captures unless CPU annotation overhead is acceptable.

## Validation result

The 128-GPU training run completed all ten requested iterations with finite
losses and gradients and no skipped or NaN iterations. Relative to the native
Mamba CP baseline, clean post-profile step time changed from 126.7454 seconds
to 90.1602 seconds. CUDA traces showed total NCCL union falling from 109.7493
seconds to 49.3296 seconds over the two-step measurement window.
