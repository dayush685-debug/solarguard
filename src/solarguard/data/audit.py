"""Walks a raw image dataset and records per-image facts: hashes, dimensions,
format, corruption status, and a modality heuristic. Produces one row per file —
no aggregation, no judgment calls — so every later analysis step works off the
same ground truth table.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from .hashing import ahash, dhash, sha256_file

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

THERMAL_NAME_HINTS = re.compile(r"(thermal|infrared|\bir\b|flir)", re.IGNORECASE)


@dataclass
class ImageRecord:
    path: str  # relative to the dataset root, forward slashes
    split: str  # as found on disk — this is the UNTRUSTED original split
    label: str
    filename: str
    file_size_bytes: int
    corrupt: bool
    corrupt_reason: str
    width: int | None
    height: int | None
    mode: str | None
    format: str | None
    grayscale_like_ratio: float | None  # fraction of sampled pixels where R==G==B
    likely_grayscale: bool
    filename_suggests_thermal: bool
    sha256: str | None
    dhash: int | None
    ahash: int | None
    suspicious: bool
    suspicious_reason: str


def _grayscale_like_ratio(img: Image.Image, sample_stride: int = 7) -> float:
    """Fraction of sampled pixels where R==G==B. 1.0 for true grayscale-content
    images even if they're technically stored in RGB mode."""
    rgb = img.convert("RGB")
    pixels = list(rgb.getdata())[::sample_stride]
    if not pixels:
        return 0.0
    equal = sum(1 for r, g, b in pixels if r == g == b)
    return equal / len(pixels)


def scan_image(root: Path, rel_path: Path, split: str, label: str) -> ImageRecord:
    full_path = root / rel_path
    file_size = full_path.stat().st_size if full_path.exists() else 0

    corrupt = False
    corrupt_reason = ""
    width = height = None
    mode = fmt = None
    gray_ratio = None
    likely_gray = False
    sha = dh = ah = None
    suspicious = False
    suspicious_reasons = []

    if not full_path.exists():
        return ImageRecord(
            path=rel_path.as_posix(), split=split, label=label, filename=rel_path.name,
            file_size_bytes=0, corrupt=True, corrupt_reason="file missing",
            width=None, height=None, mode=None, format=None,
            grayscale_like_ratio=None, likely_grayscale=False,
            filename_suggests_thermal=bool(THERMAL_NAME_HINTS.search(rel_path.name)),
            sha256=None, dhash=None, ahash=None,
            suspicious=True, suspicious_reason="file missing",
        )

    if file_size == 0:
        suspicious = True
        suspicious_reasons.append("zero-byte file")

    try:
        sha = sha256_file(full_path)
    except Exception as e:
        corrupt = True
        corrupt_reason = f"could not read bytes: {e}"

    if not corrupt:
        try:
            with Image.open(full_path) as img:
                img.verify()
            with Image.open(full_path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format
                gray_ratio = _grayscale_like_ratio(img)
                likely_gray = gray_ratio is not None and gray_ratio >= 0.95
                dh = dhash(img)
                ah = ahash(img)
        except Exception as e:
            corrupt = True
            corrupt_reason = f"PIL could not decode: {e}"

    if not corrupt:
        if width is not None and height is not None:
            if min(width, height) < 32:
                suspicious = True
                suspicious_reasons.append(f"very small dimension ({width}x{height})")
            aspect = max(width, height) / max(1, min(width, height))
            if aspect > 6:
                suspicious = True
                suspicious_reasons.append(f"extreme aspect ratio ({width}x{height})")
        if file_size < 2000:
            suspicious = True
            suspicious_reasons.append(f"very small file size ({file_size} bytes)")

    return ImageRecord(
        path=rel_path.as_posix(),
        split=split,
        label=label,
        filename=rel_path.name,
        file_size_bytes=file_size,
        corrupt=corrupt,
        corrupt_reason=corrupt_reason,
        width=width,
        height=height,
        mode=mode,
        format=fmt,
        grayscale_like_ratio=gray_ratio,
        likely_grayscale=likely_gray,
        filename_suggests_thermal=bool(THERMAL_NAME_HINTS.search(rel_path.name)),
        sha256=sha,
        dhash=dh,
        ahash=ah,
        suspicious=suspicious,
        suspicious_reason="; ".join(suspicious_reasons),
    )


def build_audit(dataset_root: Path) -> pd.DataFrame:
    """dataset_root should contain Train/Valid/Test, each containing class folders."""
    records: list[ImageRecord] = []
    for split_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        split = split_dir.name
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    rel = f.relative_to(dataset_root)
                    records.append(scan_image(dataset_root, rel, split, label))
    return pd.DataFrame([asdict(r) for r in records])
