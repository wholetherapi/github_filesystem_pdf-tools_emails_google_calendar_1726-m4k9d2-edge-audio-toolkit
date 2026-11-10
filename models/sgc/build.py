"""Graph construction for the streaming front-end."""

from models.sgc.ops import GateDensitySchedule, SparseGateConv1d


def build_frontend(layers=8, channels=96):
    schedule = GateDensitySchedule(0.75, 0.25, layers)
    return [SparseGateConv1d(channels, 5, schedule) for _ in range(layers)]
