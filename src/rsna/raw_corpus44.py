"""Recovered corpus44 raw-DICOM pixel preparation; serving remains unqualified.

Adapted from our independently implemented ``training_v2/cache_contract.py``
(reference SHA256 de974ecb69853a3874b9bb20e39b79eb78ac5ed94ed6f5b702e0ba92571dc714).
The pixel mechanics are unchanged. Dreaddevelopment's Apache-2.0 inference
notebook and TranBaDat's CORPUS44_336 recipe informed that implementation;
source URLs remain in CONTRACT and THIRD_PARTY.md, with Apache-2.0.txt retained.

Recovered from the earlier private serving source, not a newly qualified
spatial submission. The spatial candidate still requires fresh, label-free
DEVELOPMENT raw/cache parity and GPU serving qualification. This
module cannot read labels, fit a model, or publish a competition submission.
No native-slice adjacency, rotation, or laterality canonicalisation is added.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SLOTS = (
    ("Sagittal", 1, 12),
    ("Sagittal", 0, 10),
    ("Coronal", 1, 8),
    ("Coronal", 0, 6),
    ("Axial", -1, 8),
)
SLOT_BOUNDS = (0, 12, 22, 30, 36, 44)
CONTRACT = {
    "name": "dread_corpus44_slot_safe_v1",
    "cache_shape": [44, 336, 336],
    "cache_dtype": "uint8",
    "slots": [list(slot) for slot in SLOTS],
    "span": [0.15, 0.85],
    "crop_mm": 140.0,
    "percentiles": [2.0, 98.0],
    "normalization_epsilon": 1e-6,
    "resize": "cv2.INTER_AREA",
    "uint8_conversion": "multiply_255_then_truncate",
    "mask": "per_slice_pixel_sum_gt_0",
    "series_selection": "first_preferred_fluid_else_first_unused_same_plane_in_csv_order",
    "slice_order": "normal_projection_ascending_stable_glob_ties_else_InstanceNumber",
    "windows": "all_within_slot_centers_with_three_positive_mask_values",
    "max_windows": 34,
    "position": "2*(center-slot_start)/(slot_width-1)-1",
    "padding_slot_id": 5,
    "empty_study": "zero_cache_and_mask; spatial_serving_rejects_no_eligible_triplet",
    "model_size": 320,
    "model_resize": "torch_bilinear_align_corners_false",
    "model_normalization": "imagenet_mean_0.485_0.456_0.406_std_0.229_0.224_0.225",
    "source_urls": [
        "https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus",
        "https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext",
        "https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model",
        "https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection/blob/main/src/rsna_knee/mil/recipes.py",
        "https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection/blob/main/docs/experiment_ledger.md",
    ],
}


def _spec(contract: Mapping[str, Any]) -> tuple[list[tuple[str, int, int]], int, int]:
    slots = [(str(p), int(f), int(n)) for p, f, n in contract["slots"]]
    depth, height, width = map(int, contract["cache_shape"])
    if height != width or sum(n for _, _, n in slots) != depth:
        raise ValueError("Invalid square cache shape or slot widths")
    return slots, depth, height


def span_indices(n_files: int, count: int, contract: Mapping[str, Any] = CONTRACT) -> np.ndarray:
    """Rounded evenly spread source indices, including repeats on short series."""
    if n_files < 1 or count < 1:
        raise ValueError("n_files and count must be positive")
    if n_files == 1:
        return np.zeros(count, dtype=np.int64)
    span_lo, span_hi = map(float, contract["span"])
    start = int(n_files * span_lo)
    stop = max(int(n_files * span_hi) - 1, start)
    return np.linspace(start, stop, count).round().astype(np.int64)


def _fluid_flag(value: Any) -> int | None:
    # Normal 0/1 values retain the original first-match semantics. Unknown flags
    # are neither preferred true nor preferred false; same-plane fallback works.
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "t"):
            return 1
        if v in ("false", "no", "n", "f"):
            return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if np.isfinite(numeric) and numeric in (0, 1) else None


def pick_series(rows: Sequence[Mapping[str, Any]], plane: str, fluid: int,
                used: set[str]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if row["Anatomical_Plane"] == plane
                  and str(row["SeriesInstanceUID"]) not in used]
    if fluid in (0, 1):
        preferred = [row for row in candidates if _fluid_flag(row.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def order_dicom_files(series_dir: str) -> tuple[list[tuple[str, float]], float]:
    """Ascending slice-normal coordinate; preserve glob ordering for tied positions."""
    import pydicom

    records, spacing_values = [], []
    for path in glob.glob(os.path.join(series_dir, "*.dcm")):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            orientation = getattr(header, "ImageOrientationPatient", None)
            position = getattr(header, "ImagePositionPatient", None)
            if orientation is not None and position is not None and len(orientation) == 6:
                normal = np.cross(np.asarray(orientation[:3], dtype=float),
                                  np.asarray(orientation[3:], dtype=float))
                coordinate = float(np.dot(np.asarray(position, dtype=float), normal))
            else:
                coordinate = float(getattr(header, "InstanceNumber", 0) or 0)
            spacing = getattr(header, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None else 0.5
            spacing_values.append(spacing)
            records.append((coordinate, path, spacing))
        except Exception:
            # The original corpus path retained failed headers at coordinate 0.
            # Pixel decoding below will still try the file and leave it zero if
            # that also fails.
            records.append((0.0, path, 0.5))
    records.sort(key=lambda item: item[0])
    median = float(np.median(spacing_values)) if spacing_values else 0.5
    return [(path, spacing) for _, path, spacing in records], median


def read_dicom_pixels(path: str) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    dicom = pydicom.dcmread(path)
    pixels = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
    if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels


def crop_resize(pixels: np.ndarray, spacing: float,
                contract: Mapping[str, Any] = CONTRACT) -> np.ndarray:
    import cv2

    _, _, size = _spec(contract)
    height, width = pixels.shape
    side = min(int(round(float(contract["crop_mm"]) / max(spacing, 1e-3))), height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return cv2.resize(pixels[top:top + side, left:left + side], (size, size),
                      interpolation=cv2.INTER_AREA)


def prepare_dicom_study(study_uid: str, series_rows: Sequence[Mapping[str, Any]],
                        series_root: str, contract: Mapping[str, Any] = CONTRACT,
                        *, order_fn: Callable = order_dicom_files,
                        read_fn: Callable = read_dicom_pixels,
                        resize_fn: Callable = crop_resize) -> tuple[np.ndarray, np.ndarray]:
    """Return corpus-compatible uint8 volume [44,336,336] and uint8 mask [44].

    ``series_rows`` must retain the competition CSV row order. Percentiles are
    computed on full decoded selected images BEFORE physical cropping, within
    each series; duplicated sampled indices count repeatedly. Missing/failed
    slices stay in their original fixed positions as zeros.
    """
    slots, depth, size = _spec(contract)
    volume = np.zeros((depth, size, size), dtype=np.uint8)
    used, offset = set(), 0
    for plane, fluid, width in slots:
        selected = pick_series(series_rows, plane, fluid, used)
        if selected is None:
            offset += width
            continue
        uid = str(selected["SeriesInstanceUID"])
        used.add(uid)
        paths, median_spacing = order_fn(os.path.join(series_root, str(study_uid), uid))
        if not paths:
            offset += width
            continue
        images, spacings = [], []
        for pick in span_indices(len(paths), width, contract):
            path, spacing = paths[min(int(pick), len(paths) - 1)]
            try:
                images.append(read_fn(path))
                spacings.append(spacing)
            except Exception:
                images.append(None)
                spacings.append(median_spacing)
        available = [image for image in images if image is not None]
        low, high = np.percentile(np.concatenate([image.ravel() for image in available]),
                                  contract["percentiles"]) if available else (0.0, 1.0)
        for local, (image, spacing) in enumerate(zip(images, spacings)):
            if image is None:
                continue
            normalized = np.clip((image - low) / (high - low + float(contract["normalization_epsilon"])), 0, 1)
            normalized = resize_fn(normalized, spacing if spacing > 0 else median_spacing, contract)
            volume[offset + local] = (normalized * 255).astype(np.uint8)
        offset += width
    mask = (volume.reshape(depth, -1).sum(axis=1) > 0).astype(np.uint8)
    return volume, mask

