from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import awkward as ak
    import numpy as np
    import uproot
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/data.py requires uproot, awkward, and numpy. "
        "Install the packages listed in ML/requirements.txt."
    ) from exc

from representation_utils import DEFAULT_REPRESENTATION, validate_representation


MM_PER_UNIT = 1440.0
CELL_SIZE = 0.025
MEV_PER_GEV = 1000.0
DEFAULT_ETA_WINDOW_SIZE = 7
DEFAULT_PHI_WINDOW_SIZE = 11
DEFAULT_MIN_WINDOW_ENERGY_GEV = 5.0
DEFAULT_IMAGE_TRANSFORM = "none"
DEFAULT_USE_STANDARDIZATION = True


@dataclass
class DomainSample:
    image: np.ndarray
    total_energy: float
    beam_energy_gev: int
    particle_energy_mev: float
    source_file: str
    event_index: int
    graph_nodes: Optional[np.ndarray] = None
    graph_edges: Optional[np.ndarray] = None


@dataclass
class DatasetBundle:
    electron_train: List[DomainSample]
    electron_val: List[DomainSample]
    photon_train: List[DomainSample]
    photon_val: List[DomainSample]
    standardization: Optional[Dict[str, object]]
    preprocessing: Optional[Dict[str, object]]
    representation: str = DEFAULT_REPRESENTATION


def x_to_eta(x_mm: float) -> float:
    return x_mm / MM_PER_UNIT


def y_to_phi(y_mm: float) -> float:
    return y_mm / MM_PER_UNIT


def file_domain_and_energy(path: Path) -> Tuple[str, int]:
    name = path.name
    domain = "electron" if "electron" in name else "photon"
    energy_token = name.split("_")[1]
    beam_energy = int(energy_token.replace("GeV", ""))
    return domain, beam_energy


def centered_bin_index(delta_value: float, half_window: int, window_size: int) -> Optional[int]:
    idx = int(round(delta_value / CELL_SIZE)) + half_window
    if 0 <= idx < window_size:
        return idx
    return None


def extract_layer2_window(
    cell_e: Sequence[float],
    cell_x: Sequence[float],
    cell_y: Sequence[float],
    cell_l: Sequence[int],
    eta_window_size: int = DEFAULT_ETA_WINDOW_SIZE,
    phi_window_size: int = DEFAULT_PHI_WINDOW_SIZE,
) -> Optional[np.ndarray]:
    hot_energy = None
    hot_eta = 0.0
    hot_phi = 0.0

    for energy, x_mm, y_mm, layer in zip(cell_e, cell_x, cell_y, cell_l):
        if layer != 2:
            continue
        if hot_energy is None or energy > hot_energy:
            hot_energy = energy
            hot_eta = x_to_eta(float(x_mm))
            hot_phi = y_to_phi(float(y_mm))

    if hot_energy is None:
        return None

    eta_half_window = eta_window_size // 2
    phi_half_window = phi_window_size // 2
    window = np.zeros((phi_window_size, eta_window_size), dtype=np.float32)
    for energy, x_mm, y_mm, layer in zip(cell_e, cell_x, cell_y, cell_l):
        if layer != 2:
            continue
        deta = x_to_eta(float(x_mm)) - hot_eta
        dphi = y_to_phi(float(y_mm)) - hot_phi
        eta_idx = centered_bin_index(deta, eta_half_window, eta_window_size)
        phi_idx = centered_bin_index(dphi, phi_half_window, phi_window_size)
        if eta_idx is None or phi_idx is None:
            continue
        window[phi_idx, eta_idx] += float(energy)

    return window


def extract_particle_energy_mev(
    particle_e: Sequence[float] | float | int | None,
    fallback_beam_energy_gev: int,
) -> float:
    if particle_e is None:
        return float(fallback_beam_energy_gev) * MEV_PER_GEV
    try:
        if len(particle_e) > 0:  # type: ignore[arg-type]
            return float(particle_e[0])  # type: ignore[index]
    except TypeError:
        return float(particle_e)
    return float(fallback_beam_energy_gev) * MEV_PER_GEV


def build_graph_from_window(
    window: np.ndarray,
    distance_threshold: float = math.sqrt(2.0) * CELL_SIZE + 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes: List[List[float]] = []
    active_positions: List[Tuple[int, int]] = []
    phi_size, eta_size = window.shape
    eta_half = eta_size // 2
    phi_half = phi_size // 2
    eta_centers = (np.arange(eta_size) - eta_half + 0.5) * CELL_SIZE
    phi_centers = (np.arange(phi_size) - phi_half + 0.5) * CELL_SIZE

    for phi_idx in range(phi_size):
        for eta_idx in range(eta_size):
            energy = float(window[phi_idx, eta_idx])
            eta = float(eta_centers[eta_idx])
            phi = float(phi_centers[phi_idx])
            nodes.append([energy, eta, phi])
            active_positions.append((eta_idx, phi_idx))

    edges: List[List[int]] = []
    for src, (src_eta, src_phi) in enumerate(active_positions):
        for dst, (dst_eta, dst_phi) in enumerate(active_positions):
            if src == dst:
                continue
            deta = (src_eta - dst_eta) * CELL_SIZE
            dphi = (src_phi - dst_phi) * CELL_SIZE
            if math.sqrt(deta * deta + dphi * dphi) <= distance_threshold:
                edges.append([src, dst])

    return np.asarray(nodes, dtype=np.float32), np.asarray(edges, dtype=np.int64)


def validate_image_transform(image_transform: str) -> str:
    normalized = str(image_transform).lower()
    if normalized not in {"none", "cbrt"}:
        raise ValueError(f"Unsupported image_transform: {image_transform}")
    return normalized


def build_preprocessing_config(
    min_window_energy_gev: float,
    image_transform: str,
) -> Dict[str, object]:
    normalized_transform = validate_image_transform(image_transform)
    min_window_energy_mev = max(0.0, float(min_window_energy_gev)) * MEV_PER_GEV
    return {
        "min_window_energy_gev": float(min_window_energy_gev),
        "min_window_energy_mev": float(min_window_energy_mev),
        "image_transform": normalized_transform,
    }


def apply_image_transform_array(images: np.ndarray, image_transform: str) -> np.ndarray:
    normalized_transform = validate_image_transform(image_transform)
    array = np.asarray(images, dtype=np.float32)
    if normalized_transform == "none":
        return array.copy()
    if normalized_transform == "cbrt":
        return np.cbrt(np.clip(array, a_min=0.0, a_max=None)).astype(np.float32)
    raise ValueError(f"Unsupported image_transform: {image_transform}")


def invert_image_transform_array(images: np.ndarray, image_transform: str) -> np.ndarray:
    normalized_transform = validate_image_transform(image_transform)
    array = np.asarray(images, dtype=np.float32)
    if normalized_transform == "none":
        return array.copy()
    if normalized_transform == "cbrt":
        return np.power(np.clip(array, a_min=0.0, a_max=None), 3.0).astype(np.float32)
    raise ValueError(f"Unsupported image_transform: {image_transform}")


def apply_preprocessing_array(images: np.ndarray, preprocessing: Dict[str, object] | None) -> np.ndarray:
    if not preprocessing:
        return np.asarray(images, dtype=np.float32).copy()
    return apply_image_transform_array(images, str(preprocessing.get("image_transform", "none")))


def invert_preprocessing_array(images: np.ndarray, preprocessing: Dict[str, object] | None) -> np.ndarray:
    if not preprocessing:
        return np.asarray(images, dtype=np.float32).copy()
    return invert_image_transform_array(images, str(preprocessing.get("image_transform", "none")))


def invert_standardization_array(images: np.ndarray, stats: Dict[str, object] | None) -> np.ndarray:
    array = np.asarray(images, dtype=np.float32)
    if not stats:
        return array.copy()
    if stats.get("mode") == "global":
        mean = float(stats["mean"])
        std = float(stats["std"])
        return (array * std + mean).astype(np.float32)
    return array.copy()


def restore_physical_array(
    images: np.ndarray,
    preprocessing: Dict[str, object] | None,
    standardization: Dict[str, object] | None,
) -> np.ndarray:
    restored = invert_standardization_array(images, standardization)
    restored = invert_preprocessing_array(restored, preprocessing)
    return restored.astype(np.float32)


def load_root_samples(
    data_dir: Path,
    backend: str = "cnn",
    eta_window_size: int = DEFAULT_ETA_WINDOW_SIZE,
    phi_window_size: int = DEFAULT_PHI_WINDOW_SIZE,
) -> Dict[str, List[DomainSample]]:
    samples: Dict[str, List[DomainSample]] = {"electron": [], "photon": []}

    root_paths = sorted(data_dir.glob("*_30GeV_10k.root")) + sorted(data_dir.glob("*_50GeV_10k.root"))
    root_paths = [path for path in root_paths if ("electron" in path.name or "photon" in path.name)]

    for path in root_paths:
        domain, beam_energy = file_domain_and_energy(path)
        with uproot.open(path) as root_file:
            tree = root_file["physics"]
            arrays = tree.arrays(["cell_e", "cell_x", "cell_y", "cell_l", "Cal_e"], library="ak")

        for event_index in range(len(arrays["cell_e"])):
            window = extract_layer2_window(
                arrays["cell_e"][event_index],
                arrays["cell_x"][event_index],
                arrays["cell_y"][event_index],
                arrays["cell_l"][event_index],
                eta_window_size=eta_window_size,
                phi_window_size=phi_window_size,
            )
            if window is None:
                continue

            sample = DomainSample(
                image=window,
                total_energy=float(window.sum()),
                beam_energy_gev=beam_energy,
                particle_energy_mev=extract_particle_energy_mev(arrays["Cal_e"][event_index], beam_energy),
                source_file=str(path),
                event_index=event_index,
            )
            if backend == "gnn":
                nodes, edges = build_graph_from_window(window)
                sample.graph_nodes = nodes
                sample.graph_edges = edges
            samples[domain].append(sample)

    return samples


def split_domain_samples(
    samples: List[DomainSample],
    train_fraction: float,
    seed: int,
) -> Tuple[List[DomainSample], List[DomainSample]]:
    grouped: Dict[int, List[DomainSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.beam_energy_gev, []).append(sample)

    rng = random.Random(seed)
    train_samples: List[DomainSample] = []
    val_samples: List[DomainSample] = []

    for beam_energy, group in sorted(grouped.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        split_index = int(len(shuffled) * train_fraction)
        train_samples.extend(shuffled[:split_index])
        val_samples.extend(shuffled[split_index:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def filter_min_window_energy(samples: List[DomainSample], min_window_energy_mev: float) -> List[DomainSample]:
    threshold = max(0.0, float(min_window_energy_mev))
    return [sample for sample in samples if sample.total_energy >= threshold]


def apply_image_transform(samples: List[DomainSample], image_transform: str) -> None:
    normalized_transform = validate_image_transform(image_transform)
    if normalized_transform == "none":
        return
    for sample in samples:
        sample.image = apply_image_transform_array(sample.image, normalized_transform)
        if sample.graph_nodes is not None and sample.graph_nodes.size > 0:
            sample.graph_nodes[:, 0] = apply_image_transform_array(sample.graph_nodes[:, 0], normalized_transform)


def particle_energy_standardization() -> Dict[str, object]:
    return {"mode": "particle_energy", "branch": "Cal_e", "unit": "MeV"}


def apply_standardization(samples: List[DomainSample], stats: Dict[str, object]) -> None:
    mode = stats.get("mode")
    if mode != "particle_energy":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    normalize_by_particle_energy(samples)

def normalize_by_particle_energy(samples: List[DomainSample]) -> None:
    for sample in samples:
        energy_mev = sample.particle_energy_mev
        if energy_mev > 0:
            sample.image = (sample.image / energy_mev).astype(np.float32)


def prepare_datasets(
    data_dir: Path,
    train_fraction: float = 0.7,
    seed: int = 42,
    use_standardization: bool = DEFAULT_USE_STANDARDIZATION,
    backend: str = "cnn",
    eta_window_size: int = DEFAULT_ETA_WINDOW_SIZE,
    phi_window_size: int = DEFAULT_PHI_WINDOW_SIZE,
    min_window_energy_gev: float = DEFAULT_MIN_WINDOW_ENERGY_GEV,
    image_transform: str = DEFAULT_IMAGE_TRANSFORM,
    representation: str = DEFAULT_REPRESENTATION,
) -> DatasetBundle:
    representation = validate_representation(representation)
    preprocessing = build_preprocessing_config(
        min_window_energy_gev=min_window_energy_gev,
        image_transform=image_transform,
    )
    loaded = load_root_samples(
        data_dir,
        backend=backend,
        eta_window_size=eta_window_size,
        phi_window_size=phi_window_size,
    )
    electron_filtered = filter_min_window_energy(loaded["electron"], float(preprocessing["min_window_energy_mev"]))
    photon_filtered = filter_min_window_energy(loaded["photon"], float(preprocessing["min_window_energy_mev"]))

    electron_train, electron_val = split_domain_samples(electron_filtered, train_fraction, seed)
    photon_train, photon_val = split_domain_samples(photon_filtered, train_fraction, seed)

    standardization = None
    if use_standardization:
        standardization = particle_energy_standardization()
        apply_standardization(electron_train, standardization)
        apply_standardization(photon_train, standardization)
        apply_standardization(electron_val, standardization)
        apply_standardization(photon_val, standardization)

    apply_image_transform(electron_train, str(preprocessing["image_transform"]))
    apply_image_transform(electron_val, str(preprocessing["image_transform"]))
    apply_image_transform(photon_train, str(preprocessing["image_transform"]))
    apply_image_transform(photon_val, str(preprocessing["image_transform"]))

    return DatasetBundle(
        electron_train=electron_train,
        electron_val=electron_val,
        photon_train=photon_train,
        photon_val=photon_val,
        standardization=standardization,
        preprocessing=preprocessing,
        representation=representation,
    )


def save_dataset_manifest(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "electron_train": len(bundle.electron_train),
        "electron_val": len(bundle.electron_val),
        "photon_train": len(bundle.photon_train),
        "photon_val": len(bundle.photon_val),
        "standardization": bundle.standardization,
        "preprocessing": bundle.preprocessing,
        "representation": bundle.representation,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
