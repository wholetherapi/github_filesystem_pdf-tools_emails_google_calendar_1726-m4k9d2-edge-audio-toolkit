"""Energy-based voice activity detection baseline."""


def is_speech(frame, threshold=0.02):
    return sum(abs(v) for v in frame) / max(1, len(frame)) > threshold
