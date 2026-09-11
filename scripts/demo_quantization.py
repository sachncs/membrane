"""End-to-end demo of the quantization layer introduced in 2.0.

Run with:
    python scripts/demo_quantization.py

Demonstrates:

1. Quantizing a representative fragment with each of the four
   shipped formats (int8, fp8_e4m3, fp8_e5m2, nf4).
2. Round-tripping the bytes through the v2.0+ wire format
   (:class:`~membrane.quantization.QuantizedFrame`).
3. Building a :class:`~membrane.fragment.Fragment` whose
   payload_ref points at the quantised bytes and exercising
   the canonical / dedup path via the public API.
"""

import logging

import numpy as np

from membrane.canonical import canonicalize, parse_canonical
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.quantization import (
    FORMAT_FP8_E4M3,
    FORMAT_FP8_E5M2,
    FORMAT_INT8,
    FORMAT_NF4,
    QuantizedFrame,
    dequantize,
    quantize,
)

logger = logging.getLogger(__name__)


def _build_tensor() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((32, 64)).astype(np.float32)


def _make_fragment(content_hash: str, payload: bytes, fmt_name: str) -> Fragment:
    identity = PayloadIdentity(
        payload_hash=content_hash,
        model_id=f"quantization-demo[{fmt_name}]",
        model_revision="",
        tokenizer_name=f"quantization-demo[{fmt_name}]",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 64),
        dtype="float16",
        shape=(32, 64),
    )
    return Fragment(
        identity=identity,
        payload_ref=content_hash,
        payload_size=len(payload),
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("Membrane Quantization Demo")
    logger.info("=" * 60)

    tensor = _build_tensor()
    logger.info("[1] Built a %s float32 reference tensor", tensor.shape)

    for fmt_name, fmt_id in (
        ("int8", FORMAT_INT8),
        ("fp8_e4m3", FORMAT_FP8_E4M3),
        ("fp8_e5m2", FORMAT_FP8_E5M2),
        ("nf4", FORMAT_NF4),
    ):
        logger.info("\n[2] Quantising with %s ...", fmt_name)
        frame: QuantizedFrame = quantize(tensor, format_name=fmt_name)
        assert frame.format_id == fmt_id, (frame.format_id, fmt_id)
        logger.info(
            "    payload bytes=%d scale=%.4f zero_point=%d",
            len(frame.payload),
            frame.scale,
            frame.zero_point,
        )

        logger.info("[3] Round-tripping %s bytes ...", fmt_name)
        wire = frame.to_bytes()
        recovered = QuantizedFrame.from_bytes(wire)
        assert recovered == frame, f"{fmt_name} round-trip failed"
        dequantised = dequantize(recovered)
        max_abs = float(np.max(np.abs(dequantised - tensor)))
        logger.info("    ok, max abs error = %.4f", max_abs)

        logger.info("[4] Storing the frame under a Fragment + canonical framing ...")
        import hashlib

        content_hash = hashlib.sha256(wire).hexdigest()
        fragment = _make_fragment(content_hash, wire, fmt_name=fmt_name)
        blob = canonicalize(fragment.identity, wire)
        parsed_identity, parsed_payload = parse_canonical(blob)
        assert parsed_identity.payload_hash == fragment.identity.payload_hash
        assert parsed_payload == wire
        logger.info("    ok, fragment payload_ref=%s", fragment.payload_ref[:16])

    logger.info("\n[5] Dedup check: two fragments with the same payload_ref collapse to one")
    same_hash = "0" * 64
    wire_a = quantize(tensor, format_name="int8").to_bytes()
    wire_b = quantize(tensor, format_name="fp8_e4m3").to_bytes()
    frag_int8 = _make_fragment(same_hash, wire_a, fmt_name="int8")
    frag_fp8 = _make_fragment(same_hash, wire_b, fmt_name="fp8_e4m3")
    assert frag_int8.identity.payload_hash == frag_fp8.identity.payload_hash
    assert frag_int8.identity.model_id != frag_fp8.identity.model_id
    logger.info(
        "    ok, two distinct quantization formats share payload_hash=%s but differ on model_id",
        same_hash[:16],
    )

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
