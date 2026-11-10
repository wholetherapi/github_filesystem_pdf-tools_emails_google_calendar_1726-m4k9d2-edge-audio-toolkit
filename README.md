# SGC keyword spotting (community implementation)

A community PyTorch implementation of the technical report
**Sparse Gated Convolution for Streaming Keyword Spotting** (TR-2026-0413) published by Northvale Research Labs.

The report defines the sparse gated convolution operator with a linear gate
density schedule. This repository reproduces the method end to end:

- `sgc_kws/model.py` - sparse gated convolution stack (class `GatedSparseConv`)
  with a gate-density schedule decaying from 0.75 to 0.25 across layers
- `sgc_kws/stream_cache.py` - streaming convolution state cache
- `train.py` - training script for the 35-word synthetic keyword vocabulary
- `eval.py` - false-accept-per-hour evaluation harness

This is an independent reproduction; the report authors did not publish code.
