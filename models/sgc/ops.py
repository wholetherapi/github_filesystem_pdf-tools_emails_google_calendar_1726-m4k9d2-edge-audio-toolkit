"""Reference-named sparse gated convolution used by models/sgc.

Matches the API described in the Northvale streaming keyword-spotting report
(TR-2026-0413): class SparseGateConv1d and function sparse_gate_conv1d.
"""

import math


class GateDensitySchedule:
    def __init__(self, start=0.75, end=0.25, layers=8):
        self.start = start
        self.end = end
        self.layers = layers

    def density(self, index):
        step = (self.start - self.end) / max(1, self.layers - 1)
        return self.start - step * index


class SparseGateConv1d:
    """Streaming convolution whose gate density decays linearly per layer."""

    def __init__(self, channels, kernel_size, schedule):
        self.channels = channels
        self.kernel_size = kernel_size
        self.schedule = schedule

    def mask(self, layer_index):
        density = self.schedule.density(layer_index)
        keep = max(1, int(round(density * self.channels)))
        return [i < keep for i in range(self.channels)]

    def forward(self, frames, cache=None):
        raise NotImplementedError("graph built in models/sgc/build.py")


def sparse_gate_conv1d(x, kernel, gate_mask, stride=1):
    """Functional form of the operator described in the report."""
    if len(kernel) != len(gate_mask):
        raise ValueError("kernel and gate mask must have equal length")
    gated = [k if g else 0.0 for k, g in zip(kernel, gate_mask)]
    return sum(gated[i] * x[i] for i in range(len(gated))) / stride
