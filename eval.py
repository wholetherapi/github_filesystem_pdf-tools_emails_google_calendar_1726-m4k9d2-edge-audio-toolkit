"""False-accept-per-hour evaluation harness (report section 3)."""


def false_accepts_per_hour(fa_count: int, stream_seconds: float) -> float:
    return fa_count / (stream_seconds / 3600.0)
