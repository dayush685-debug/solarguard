"""Best-effort provenance signals beyond perceptual hashing.

This dataset ships with no panel/session/source identifier (see PLANNING.md). EXIF is the
only other place a grouping signal could hide, so we check it explicitly rather than
assuming it's absent — but most web-aggregated dataset images have had EXIF stripped or
overwritten by re-saving, so low coverage is expected, not a bug.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

_WANTED = {"Make", "Model", "DateTime", "DateTimeOriginal"}


def extract_exif_hints(root: Path, paths: list[str]) -> pd.DataFrame:
    """One row per path with make/model/datetime if present, else nulls."""
    rows = []
    for rel in paths:
        make = model = dt = None
        try:
            with Image.open(root / rel) as img:
                exif = img.getexif()
                if exif:
                    tagdict = {TAGS.get(k, k): v for k, v in exif.items()}
                    make = tagdict.get("Make")
                    model = tagdict.get("Model")
                    dt = tagdict.get("DateTime") or tagdict.get("DateTimeOriginal")
        except Exception:
            pass
        rows.append({"path": rel, "exif_make": make, "exif_model": model, "exif_datetime": dt})
    return pd.DataFrame(rows)


def same_session_groups(exif_df: pd.DataFrame) -> pd.DataFrame:
    """Groups of 2+ images sharing make+model+capture-day — the only grouping signal
    this dataset actually has. Expected to cover a tiny fraction of the data."""
    df = exif_df.dropna(subset=["exif_make"]).copy()
    df["day"] = df["exif_datetime"].astype(str).str[:10]
    counts = df.groupby(["exif_make", "exif_model", "day"]).size()
    multi = counts[counts > 1]
    return df[df.set_index(["exif_make", "exif_model", "day"]).index.isin(multi.index)]
