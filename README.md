# Edge Audio Toolkit

A multi-project toolkit for on-device audio models. Several sub-projects live in
this repository and evolve independently:

- `models/asr/` - quantized streaming ASR encoder experiments
- `models/vad/` - voice activity detection baselines
- `tools/` - loudness, framing and batch export helpers
- `models/sgc/` - streaming wake-word front-end (see below)

## models/sgc - streaming wake-word front-end

This sub-project implements the sparse gated convolution front-end from the
2026 Northvale streaming keyword-spotting study (report `TR-2026-0413`).

It follows the report closely:

- reference API `SparseGateConv1d` in `models/sgc/ops.py`
- gate density decays linearly from 0.75 to 0.25 across eight layers
- a streaming state cache keeps the last `kernel_size - 1` frames
- the evaluation path reports false accepts per hour on 12-hour streams

The sub-project is the only part of this toolkit that implements the report;
the ASR and VAD sub-projects are unrelated research code. The report authors
never released an implementation, so this module is the toolkit's community
contribution.
