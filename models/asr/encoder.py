"""Quantized streaming ASR encoder experiment (unrelated to models/sgc)."""


def encoder_config():
    return {"layers": 12, "hidden": 256, "quant": "int8"}
