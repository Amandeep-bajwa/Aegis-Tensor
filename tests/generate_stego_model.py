"""Generate deterministic clean and synthetic steganography test models.

The clean weights are normally distributed, the way trained weights look. The
injected tensors are:

* ``layer_1.weight``: the whole tensor overwritten with random bytes (naive blob).
* ``layer_2.weight``: a readable text payload in the least significant mantissa
  bit of every weight (structured LSB payload).
* ``layer_3.weight``: a random payload in the least significant mantissa bit of
  every weight. Random low bits are indistinguishable from trained weights, so
  this case is expected to pass the static scan. ``aegis disarm`` removes it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


MODEL_SIZE = 256
TEXT_PAYLOAD = b"Aegis-Tensor synthetic steganography payload: connect 10.0.0.5:4444 " * 4000


def _clean_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        f"layer_{index}.weight": rng.normal(
            loc=0.0, scale=0.02, size=(MODEL_SIZE, MODEL_SIZE)
        ).astype(np.float32)
        for index in range(1, 6)
    }


def _naive_injection(rng: np.random.Generator) -> np.ndarray:
    """Replace every float with random bytes (encrypted-blob style)."""
    payload = rng.integers(0, 256, size=MODEL_SIZE * MODEL_SIZE * 4, dtype=np.uint8)
    return payload.view(np.float32).reshape(MODEL_SIZE, MODEL_SIZE)


def _set_lsb(weights: np.ndarray, bits: np.ndarray) -> np.ndarray:
    words = weights.reshape(-1).view(np.uint32).copy()
    words = (words & ~np.uint32(1)) | bits.astype(np.uint32)
    return words.view(np.float32).reshape(weights.shape)


def _text_lsb_injection(weights: np.ndarray) -> np.ndarray:
    """Write a readable text payload into the least significant mantissa bit."""
    count = weights.size
    payload_bits = np.unpackbits(np.frombuffer(TEXT_PAYLOAD, dtype=np.uint8), bitorder="little")
    return _set_lsb(weights, np.resize(payload_bits, count))


def _random_lsb_injection(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Write random bits into the least significant mantissa bit."""
    return _set_lsb(weights, rng.integers(0, 2, size=weights.size, dtype=np.uint32))


def generate_models(output_dir: Path, seed: int = 20260922) -> tuple[Path, Path]:
    """Write clean_model.safetensors and infected_model.safetensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clean = _clean_weights(rng)
    infected = dict(clean)
    infected["layer_1.weight"] = _naive_injection(rng)
    infected["layer_2.weight"] = _text_lsb_injection(clean["layer_2.weight"])
    infected["layer_3.weight"] = _random_lsb_injection(clean["layer_3.weight"], rng)

    clean_path = output_dir / "clean_model.safetensors"
    infected_path = output_dir / "infected_model.safetensors"
    save_file(clean, str(clean_path), metadata={"generator": "aegis-stego-test"})
    save_file(infected, str(infected_path), metadata={"generator": "aegis-stego-test"})
    return clean_path, infected_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "fixtures",
        help="Directory for generated safetensors files.",
    )
    args = parser.parse_args()
    clean_path, infected_path = generate_models(args.output_dir)
    print(f"Generated {clean_path}")
    print(f"Generated {infected_path}")


if __name__ == "__main__":
    main()
