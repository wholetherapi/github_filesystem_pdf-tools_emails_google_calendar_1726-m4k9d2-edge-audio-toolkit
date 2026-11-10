"""Sparse gated convolution stack (community implementation of TR-2026-0413)."""

import math


class GateDensitySchedule:
    """Linear decay of gate density across layers (report section 2.2)."""

    def __init__(self, start=0.75, end=0.25, layers=8):
        self.start = start
        self.end = end
        self.layers = layers

    def density(self, layer_index: int) -> float:
        step = (self.start - self.end) / max(1, self.layers - 1)
        return self.start - step * layer_index


class GatedSparseConv:
    """Dense-gated convolution with sparse gate masks.

    Naming differs from the reference API on purpose: this project uses its own
    class names, the maths follows the report.
    """

    def __init__(self, in_channels, out_channels, kernel_size, schedule):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.schedule = schedule

    def gate_mask(self, layer_index: int):
        density = self.schedule.density(layer_index)
        count = max(1, int(round(density * self.out_channels)))
        return [i < count for i in range(self.out_channels)]

    def forward(self, frames, state=None):
        # Streaming convolution with a compact gate mask.
        raise NotImplementedError("reference sketch - see train.py for the full graph")
