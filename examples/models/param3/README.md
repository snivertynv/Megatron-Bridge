# Param3

This directory contains a Slurm launcher for the Param3 74B training recipe.
The recipe reconstructs the architecture and training settings captured in
`fusedmhc_flashqla_TP2PP1EP811030.out`:

- 40 decoder layers with a repeating full-attention/GatedDeltaNet pattern
- mHC hyper-connections
- one dense layer followed by 39 MoE layers with 128 experts
- TP=2, PP=1, EP=8 on 32 GPUs
- mock data, global batch size 64, and random initialization
- cuTeDSL fused routed-expert MLPs; CUDA graphs remain disabled

The launcher requires a Megatron-Core build containing the Param3 mHC support
used by the original run. Set `CONTAINER_IMAGE`, then submit it with:

```bash
sbatch examples/models/param3/slurm_pretrain.sh
```

The launcher selects
`param3_74b_pretrain_32gpu_h100_bf16_cutedsl_config` by default. To run the
unfused parity/debug recipe instead:

```bash
sbatch --export=ALL,RECIPE_NAME=param3_74b_pretrain_32gpu_h100_bf16_config \
  examples/models/param3/slurm_pretrain.sh
```

The supplied run log does not expose Engram configuration or a separate sparse
attention implementation. GatedDeltaNet is linear attention, not token-sparse
attention, so those features are intentionally not inferred here.
