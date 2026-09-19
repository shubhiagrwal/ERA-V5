# ZeRO Benchmark Analysis

**The Core Problem: Data Parallelism is Redundant**

In standard Distributed Data Parallel (DDP), data processing is highly efficient:  
For 32 GPUs -> process 32 unique batches of data simultaneously.

However, the memory architecture is massively redundant. To keep the math consistent, every single GPU stores a 100% complete copy of the model parameters, the gradients, and the Adam optimizer states. Because Adam stores momentum and variance variables in FP32, the optimizer states alone consume 3x to 12x more VRAM than the model itself.

ZeRO fixes this by treating the model's memory footprint like a 32-piece puzzle, assigning "ownership" of different pieces to different GPUs.

## Simulator results & benchmark analysis:

### 1. Correctness Check:

```text
CORRECTNESS CHECK
max|d_loss| = 0.000e+00 PASS
```

ZeRO is strictly a memory management architecture. It does not compress, quantize, or approximate the math. The final gradients and loss values match standard DDP down to the decimal.

### 2. The network cost model(zero ½ are “free”):

```text
COMM COST MODEL CHECK
All-Reduce (Standard DDP)      : 3875.0 bytes/GPU
Reduce-Scatter + Gather (ZeRO) : 3875.0 bytes/GPU
```

**How sharding impact network overhead?**

The simulator proves that mathematically, the volume of data sent during a full `All-Reduce` is identical to splitting the operation into a `Reduce-Scatter` (Phase 2) and an `All-Gather` (Phase 4). ZeRO-1 and 2 gives memory savings for zero extra network cost.

### 3. VRAM vs Network Trade-offs:

```text
TRAINING RUNS (Peak VRAM & Network Vol)
Baseline DDP : 51.09 MB VRAM | 12.31 MB Network
ZeRO-1 / 2   : 14.17 MB VRAM | 12.31 MB Network
ZeRO-3       :  8.02 MB VRAM | 406.09 MB Network
```

Zero-½: drops peak memory from ~51MB to ~14 MB while keeping network communication identical to the baseline.

**ZeRO-3** achieves the ultimate VRAM reduction (8.02 MB) but requires a **~33x spike in network communication** because GPUs are constantly broadcasting parameter weights back and forth during the sequential layer-by-layer forward and backward passes.

### 4. Intra-Step memory timelines(gpu 0 )

The intra-step trace reveals the exact lifecycle of a step on GPU 0:

- **In ZeRO-1:** The `optimizer_resident` memory drops from the baseline's 44.46 MB to just 7.54 MB.
- **In ZeRO-2:** `after_reduce_scatter` footprint drop instantly to 8.02 MB before the optimizer step, proving the on-the-fly deletion of raw gradients during the backward pass.
- **In ZeRO-3:** The `params_resident` at rest is a tiny **0.20 MB**. It only spikes to 8.02 MB during the `after_fwd_bwd` phase because the GPU is temporarily borrowing parameter slices from the rest of the cluster to process its data batch.
