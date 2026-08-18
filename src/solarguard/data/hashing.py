"""Exact and perceptual image hashing used for duplicate/near-duplicate detection.

No external perceptual-hashing library is used — dHash and aHash are implemented
directly against Pillow so the exact algorithm is auditable rather than opaque.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageFile

# Some JPEGs in this dataset are slightly truncated; without this, PIL raises
# on load instead of letting us flag the file as suspicious ourselves.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Exact-duplicate signature: hash of the raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash: robust to recompression/minor edits, sensitive to content changes."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            bits <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                bits |= 1
    return bits


def ahash(image: Image.Image, hash_size: int = 8) -> int:
    """Average hash: simpler, used only as a cross-check against dHash."""
    gray = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p > avg else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
