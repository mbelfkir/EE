from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/dataset.py requires torch and numpy. Install the packages listed in ML/requirements.txt."
    ) from exc


ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

from data import DatasetBundle, DomainSample, invert_preprocessing_array, prepare_datasets, save_dataset_manifest  # noqa: E402
from data import DEFAULT_ETA_WINDOW_SIZE, DEFAULT_IMAGE_TRANSFORM, DEFAULT_MIN_WINDOW_ENERGY_GEV, DEFAULT_PHI_WINDOW_SIZE, DEFAULT_USE_STANDARDIZATION  # noqa: E402
from representation_utils import compute_reta_rphi, sample_to_representation, validate_representation  # noqa: E402


def compute_normalization_scale(
    sample_sets: Sequence[Sequence[DomainSample]],
    symmetric_range: bool = False,
) -> float:
    maximum = 0.0
    for sample_set in sample_sets:
        for sample in sample_set:
            values = np.asarray(sample.image, dtype=np.float32)
            candidate = float(np.max(np.abs(values))) if symmetric_range else float(np.max(values))
            maximum = max(maximum, candidate)
    if maximum <= 0.0:
        return 1.0
    return maximum


def validate_window_shape(window: np.ndarray, phi_window_size: int, eta_window_size: int) -> None:
    expected_shape = (phi_window_size, eta_window_size)
    if tuple(window.shape) != expected_shape:
        raise ValueError(f"Expected window shape {expected_shape}, got {tuple(window.shape)}")


def normalize_window(
    window: np.ndarray,
    normalization_scale: float,
    phi_window_size: int,
    eta_window_size: int,
    symmetric_range: bool = False,
) -> Tensor:
    validate_window_shape(window, phi_window_size, eta_window_size)
    if symmetric_range:
        clipped = np.clip(window.astype(np.float32), -normalization_scale, normalization_scale)
        normalized = clipped / normalization_scale
    else:
        clipped = np.clip(window.astype(np.float32), 0.0, normalization_scale)
        normalized = 2.0 * (clipped / normalization_scale) - 1.0
    return torch.from_numpy(normalized).unsqueeze(0)


def condition_bounds(sample_sets: Sequence[Sequence[DomainSample]]) -> tuple[float, float, float, float]:
    reta_values: list[float] = []
    rphi_values: list[float] = []
    for sample_set in sample_sets:
        for sample in sample_set:
            reta, rphi = compute_reta_rphi(sample.image)
            reta_values.append(float(reta))
            rphi_values.append(float(rphi))
    if not reta_values or not rphi_values:
        return 0.0, 1.0, 0.0, 1.0
    return (
        float(min(reta_values)),
        float(max(reta_values)),
        float(min(rphi_values)),
        float(max(rphi_values)),
    )


def normalize_condition(sample: DomainSample, bounds: tuple[float, float, float, float]) -> np.ndarray:
    reta, rphi = compute_reta_rphi(sample.image)
    reta_min, reta_max, rphi_min, rphi_max = bounds
    reta_den = max(reta_max - reta_min, 1e-6)
    rphi_den = max(rphi_max - rphi_min, 1e-6)
    return np.asarray(
        [
            float((reta - reta_min) / reta_den),
            float((rphi - rphi_min) / rphi_den),
        ],
        dtype=np.float32,
    )


def sample_identifier(sample: DomainSample) -> str:
    return f"{Path(sample.source_file).stem}_event{sample.event_index:06d}"


def particle_energy_scale_mev(sample: DomainSample) -> float:
    return max(float(sample.particle_energy_mev), 1.0)


def normalize_voxel_representation(sample: DomainSample) -> np.ndarray:
    voxel = sample_to_representation(sample, representation="voxel").astype(np.float32, copy=True)
    voxel[:, 0] /= particle_energy_scale_mev(sample)
    return voxel


class UnpairedCalorimeterDataset(Dataset):
    def __init__(
        self,
        electron_samples: Sequence[DomainSample],
        photon_samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        representation: str = "image",
        condition_range: tuple[float, float, float, float] | None = None,
        symmetric_range: bool = False,
        match_beam_energy: bool = True,
        serial_batches: bool = False,
    ) -> None:
        self.electron_samples = list(electron_samples)
        self.photon_samples = list(photon_samples)
        if not self.electron_samples or not self.photon_samples:
            raise ValueError("Both electron and photon sample lists must be non-empty.")

        self.normalization_scale = float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        self.representation = validate_representation(representation)
        self.condition_range = condition_range if condition_range is not None else condition_bounds([self.electron_samples, self.photon_samples])
        self.symmetric_range = bool(symmetric_range)
        self.match_beam_energy = bool(match_beam_energy)
        self.serial_batches = serial_batches
        self.photon_by_beam = {}
        for sample in self.photon_samples:
            self.photon_by_beam.setdefault(int(sample.beam_energy_gev), []).append(sample)

    def __len__(self) -> int:
        return max(len(self.electron_samples), len(self.photon_samples))

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        electron_sample = self.electron_samples[index % len(self.electron_samples)]
        if self.serial_batches:
            photon_sample = self.photon_samples[index % len(self.photon_samples)]
        elif self.match_beam_energy and int(electron_sample.beam_energy_gev) in self.photon_by_beam:
            beam_group = self.photon_by_beam[int(electron_sample.beam_energy_gev)]
            photon_sample = beam_group[random.randrange(len(beam_group))]
        else:
            photon_sample = self.photon_samples[random.randrange(len(self.photon_samples))]

        if self.representation == "image":
            a_tensor = normalize_window(
                electron_sample.image,
                self.normalization_scale,
                self.phi_window_size,
                self.eta_window_size,
                symmetric_range=self.symmetric_range,
            )
            b_tensor = normalize_window(
                photon_sample.image,
                self.normalization_scale,
                self.phi_window_size,
                self.eta_window_size,
                symmetric_range=self.symmetric_range,
            )
        else:
            a_tensor = torch.from_numpy(normalize_voxel_representation(electron_sample))
            b_tensor = torch.from_numpy(normalize_voxel_representation(photon_sample))
        return {
            "A": a_tensor,
            "B": b_tensor,
            "A_cond": torch.from_numpy(normalize_condition(electron_sample, self.condition_range)),
            "B_cond": torch.from_numpy(normalize_condition(photon_sample, self.condition_range)),
            "A_energy_scale": torch.tensor([particle_energy_scale_mev(electron_sample)], dtype=torch.float32),
            "B_energy_scale": torch.tensor([particle_energy_scale_mev(photon_sample)], dtype=torch.float32),
            "A_path": sample_identifier(electron_sample),
            "B_path": sample_identifier(photon_sample),
        }


class SingleDomainCalorimeterDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        representation: str = "image",
        condition_range: tuple[float, float, float, float] | None = None,
        symmetric_range: bool = False,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Input sample list must be non-empty.")
        self.normalization_scale = float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        self.representation = validate_representation(representation)
        self.condition_range = condition_range if condition_range is not None else condition_bounds([self.samples])
        self.symmetric_range = bool(symmetric_range)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        if self.representation == "image":
            tensor = normalize_window(
                sample.image,
                self.normalization_scale,
                self.phi_window_size,
                self.eta_window_size,
                symmetric_range=self.symmetric_range,
            )
        else:
            tensor = torch.from_numpy(normalize_voxel_representation(sample))
        return {
            "image": tensor,
            "cond": torch.from_numpy(normalize_condition(sample, self.condition_range)),
            "energy_scale": torch.tensor([particle_energy_scale_mev(sample)], dtype=torch.float32),
            "path": sample_identifier(sample),
        }


class ElectronPhotonClassificationDataset(Dataset):
    def __init__(
        self,
        electron_samples: Sequence[DomainSample],
        photon_samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        symmetric_range: bool = False,
    ) -> None:
        self.examples: list[tuple[DomainSample, float, str]] = [
            (sample, 0.0, "electron")
            for sample in electron_samples
        ] + [
            (sample, 1.0, "photon")
            for sample in photon_samples
        ]
        if not self.examples:
            raise ValueError("Classification dataset must contain at least one example.")
        self.normalization_scale = float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        self.symmetric_range = bool(symmetric_range)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample, label, domain = self.examples[index]
        tensor = normalize_window(
            sample.image,
            self.normalization_scale,
            self.phi_window_size,
            self.eta_window_size,
            symmetric_range=self.symmetric_range,
        )
        return {
            "image": tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "domain": domain,
            "path": sample_identifier(sample),
        }


def get_split_samples(bundle: DatasetBundle, split: str) -> tuple[Sequence[DomainSample], Sequence[DomainSample]]:
    if split == "train":
        return bundle.electron_train, bundle.photon_train
    if split in {"val", "test"}:
        return bundle.electron_val, bundle.photon_val
    raise ValueError(f"Unsupported split: {split}")
