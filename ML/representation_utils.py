from __future__ import annotations

from typing import Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("ML/representation_utils.py requires numpy.") from exc


CELL_SIZE = 0.025
DEFAULT_REPRESENTATION = "image"
VOXEL_FEATURE_NAMES = ("energy", "delta_eta", "delta_phi", "r", "alpha")


def validate_representation(representation: str) -> str:
    normalized = str(representation).lower()
    if normalized not in {"image", "voxel"}:
        raise ValueError(f"Unsupported representation: {representation}")
    return normalized


def window_offset_grids(phi_window_size: int, eta_window_size: int) -> tuple[np.ndarray, np.ndarray]:
    eta_offsets = (np.arange(eta_window_size, dtype=np.float32) - (eta_window_size // 2)) * CELL_SIZE
    phi_offsets = (np.arange(phi_window_size, dtype=np.float32) - (phi_window_size // 2)) * CELL_SIZE
    delta_eta = np.broadcast_to(eta_offsets[None, :], (phi_window_size, eta_window_size)).astype(np.float32)
    delta_phi = np.broadcast_to(phi_offsets[:, None], (phi_window_size, eta_window_size)).astype(np.float32)
    return delta_eta, delta_phi


def sum_window(window: np.ndarray, eta_bins: int, phi_bins: int) -> float:
    image = np.asarray(window, dtype=np.float32)
    center_eta = image.shape[1] // 2
    center_phi = image.shape[0] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_start = center_eta - eta_half
    eta_stop = center_eta + eta_half + 1
    phi_start = center_phi - phi_half
    phi_stop = center_phi + phi_half + 1
    return float(image[phi_start:phi_stop, eta_start:eta_stop].sum())


def compute_reta_rphi(window: np.ndarray, cal_e: float) -> np.ndarray:
    image = np.asarray(window, dtype=np.float32) * cal_e # rescale back to total energy
    e7x7 = sum_window(image, 7, 7)
    e3x7 = sum_window(image, 3, 7)
    e3x3 = sum_window(image, 3, 3)
    reta = 0.0 if e7x7 <= 0.0 else e3x7 / e7x7
    rphi = 0.0 if e3x7 <= 0.0 else e3x3 / e3x7
    return np.asarray([reta, rphi], dtype=np.float32)


def window_to_voxel(window: np.ndarray) -> np.ndarray:
    image = np.asarray(window, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"window_to_voxel expects a 2D [phi, eta] window, got shape {image.shape}")
    phi_window_size, eta_window_size = image.shape
    delta_eta, delta_phi = window_offset_grids(phi_window_size, eta_window_size)
    radius = np.sqrt(delta_eta ** 2 + delta_phi ** 2).astype(np.float32)
    alpha = np.arctan2(delta_phi, delta_eta).astype(np.float32)
    voxel = np.stack([image, delta_eta, delta_phi, radius, alpha], axis=-1)
    return voxel.reshape(-1, len(VOXEL_FEATURE_NAMES)).astype(np.float32)


def voxel_to_image(voxel: np.ndarray, phi_window_size: int, eta_window_size: int) -> np.ndarray:
    array = np.asarray(voxel, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] == len(VOXEL_FEATURE_NAMES):
            energy = array[:, 0]
        else:
            energy = array.reshape(-1)
        return energy.reshape(phi_window_size, eta_window_size).astype(np.float32)
    if array.ndim == 1:
        return array.reshape(phi_window_size, eta_window_size).astype(np.float32)
    raise ValueError(f"voxel_to_image expects shape [num_voxels] or [num_voxels, {len(VOXEL_FEATURE_NAMES)}], got {array.shape}")


def window_to_representation(window: np.ndarray, representation: str = DEFAULT_REPRESENTATION) -> np.ndarray:
    representation = validate_representation(representation)
    if representation == "image":
        return np.asarray(window, dtype=np.float32).copy()
    return window_to_voxel(window)


def sample_to_representation(sample: Mapping[str, object] | object, representation: str = DEFAULT_REPRESENTATION) -> np.ndarray:
    image = getattr(sample, "image", None)
    if image is None and isinstance(sample, Mapping):
        image = sample["image"]
    if image is None:
        raise ValueError("sample_to_representation requires an object or mapping with an 'image' entry.")
    return window_to_representation(np.asarray(image, dtype=np.float32), representation=representation)


def stack_sample_representations(
    samples: Sequence[Mapping[str, object] | object],
    representation: str = DEFAULT_REPRESENTATION,
) -> np.ndarray:
    return np.stack([sample_to_representation(sample, representation=representation) for sample in samples], axis=0).astype(np.float32)


def stack_window_representations(
    windows: np.ndarray,
    representation: str = DEFAULT_REPRESENTATION,
) -> np.ndarray:
    arrays = [window_to_representation(window, representation=representation) for window in np.asarray(windows, dtype=np.float32)]
    return np.stack(arrays, axis=0).astype(np.float32)


def mean_voxel_representation(samples: Sequence[Mapping[str, object] | object]) -> np.ndarray:
    voxels = stack_sample_representations(samples, representation="voxel")
    mean_voxel = voxels[0].copy()
    mean_voxel[:, 0] = voxels[:, :, 0].mean(axis=0)
    return mean_voxel.astype(np.float32)


def voxel_rows(voxel: np.ndarray) -> list[dict[str, float | int]]:
    array = np.asarray(voxel, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(VOXEL_FEATURE_NAMES):
        raise ValueError(f"voxel_rows expects shape [num_voxels, {len(VOXEL_FEATURE_NAMES)}], got {array.shape}")
    rows: list[dict[str, float | int]] = []
    for flat_index, item in enumerate(array):
        rows.append(
            {
                "flat_index": flat_index,
                "energy": float(item[0]),
                "delta_eta": float(item[1]),
                "delta_phi": float(item[2]),
                "r": float(item[3]),
                "alpha": float(item[4]),
            }
        )
    return rows
