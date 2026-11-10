"""Training entry point for the 35-word synthetic keyword vocabulary."""

from sgc_kws.model import GateDensitySchedule, GatedSparseConv


def build_model(layers=8):
    schedule = GateDensitySchedule(start=0.75, end=0.25, layers=layers)
    return [GatedSparseConv(64, 64, 5, schedule) for _ in range(layers)]


if __name__ == "__main__":
    model = build_model()
    print(f"built {len(model)} sparse gated layers")
