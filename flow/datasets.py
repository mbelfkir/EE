from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from flow.utils import domain_to_index, ensure_ml_path


ensure_ml_path()

from data import (  # noqa: E402
    DEFAULT_ETA_WINDOW_SIZE,
    DEFAULT_IMAGE_TRANSFORM,
    DEFAULT_MIN_WINDOW_ENERGY_GEV,
    DEFAULT_PHI_WINDOW_SIZE,
    DEFAULT_USE_STANDARDIZATION,
    DatasetBundle,
    DomainSample,
    prepare_datasets,
)
from representation_utils import compute_reta_rphi  # noqa: E402


IMAGE_SCALING_MODE = "divide_by_cal_e"


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


def normalize_window(
    window: np.ndarray,
    normalization_scale: float,
    phi_window_size: int,
    eta_window_size: int,
    symmetric_range: bool = False,
) -> Tensor:
    del normalization_scale, phi_window_size, eta_window_size, symmetric_range
    return torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0)


def condition_bounds(sample_sets: Sequence[Sequence[DomainSample]]) -> tuple[float, float, float, float]:
    reta_values: list[float] = []
    rphi_values: list[float] = []
    for sample_set in sample_sets:
        for sample in sample_set:
            reta, rphi = compute_reta_rphi(sample.image, sample.particle_energy_mev)
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
    reta, rphi = compute_reta_rphi(sample.image, sample.particle_energy_mev)
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


def get_split_samples(bundle: DatasetBundle, split: str) -> tuple[Sequence[DomainSample], Sequence[DomainSample]]:
    if split == "train":
        return bundle.electron_train, bundle.photon_train
    if split in {"val", "test"}:
        return bundle.electron_val, bundle.photon_val
    raise ValueError(f"Unsupported split: {split}")


def select_fixed_samples(samples: Sequence[DomainSample], max_count: int, seed: int) -> list[DomainSample]:
    items = list(samples)
    if max_count <= 0 or len(items) <= max_count:
        return items
    indices = sorted(random.Random(seed).sample(range(len(items)), max_count))
    return [items[index] for index in indices]


def samples_to_array(samples: Sequence[DomainSample]) -> np.ndarray:
    if not samples:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.stack([np.asarray(sample.image, dtype=np.float32) for sample in samples], axis=0).astype(np.float32)


def save_dataset_manifest(bundle: DatasetBundle, output_dir: str | Path) -> None:
    """Write a flow-specific manifest without misleading legacy standardization stats."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "electron_train": len(bundle.electron_train),
        "electron_val": len(bundle.electron_val),
        "photon_train": len(bundle.photon_train),
        "photon_val": len(bundle.photon_val),
        "preprocessing": bundle.preprocessing,
        "representation": bundle.representation,
        "image_scaling": IMAGE_SCALING_MODE,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))


class UnpairedCalorimeterDataset(Dataset):
    """Unpaired image dataset built on the shared ML data contract."""

    def __init__(
        self,
        electron_samples: Sequence[DomainSample],
        photon_samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        condition_range: tuple[float, float, float, float] | None = None,
        symmetric_range: bool = False,
        match_beam_energy: bool = True,
        serial_batches: bool = False,
    ) -> None:
        self.electron_samples = list(electron_samples)
        self.photon_samples = list(photon_samples)
        if not self.electron_samples or not self.photon_samples:
            raise ValueError("Both electron and photon sample lists must be non-empty.")

        self.normalization_scale = 0.0 if normalization_scale is None else float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        self.condition_range = (
            condition_range
            if condition_range is not None
            else condition_bounds([self.electron_samples, self.photon_samples])
        )
        self.symmetric_range = bool(symmetric_range)
        self.match_beam_energy = bool(match_beam_energy)
        self.serial_batches = bool(serial_batches)
        self.photon_by_beam: dict[int, list[DomainSample]] = {}
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
    """Single-domain view of the simple-pipeline calorimeter samples."""

    def __init__(
        self,
        samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        condition_range: tuple[float, float, float, float] | None = None,
        symmetric_range: bool = False,
        domain_name: str | None = None,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Input sample list must be non-empty.")
        self.normalization_scale = 0.0 if normalization_scale is None else float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        self.condition_range = condition_range if condition_range is not None else condition_bounds([self.samples])
        self.symmetric_range = bool(symmetric_range)
        self.domain_name = domain_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        item: dict[str, Tensor | str] = {
            "image": normalize_window(
                sample.image,
                self.normalization_scale,
                self.phi_window_size,
                self.eta_window_size,
                symmetric_range=self.symmetric_range,
            ),
            "cond": torch.from_numpy(normalize_condition(sample, self.condition_range)),
            "energy_scale": torch.tensor([particle_energy_scale_mev(sample)], dtype=torch.float32),
            "path": sample_identifier(sample),
        }
        if self.domain_name is not None:
            item["domain"] = self.domain_name
            item["domain_index"] = torch.tensor(domain_to_index(self.domain_name), dtype=torch.long)
        return item


class CombinedCalorimeterDataset(Dataset):
    """Joint electron+photon autoencoder dataset with domain metadata."""

    def __init__(
        self,
        electron_samples: Sequence[DomainSample],
        photon_samples: Sequence[DomainSample],
        normalization_scale: float,
        phi_window_size: int,
        eta_window_size: int,
        condition_range: tuple[float, float, float, float] | None = None,
        symmetric_range: bool = False,
    ) -> None:
        self.samples: list[tuple[str, DomainSample]] = [
            ("electron", sample) for sample in electron_samples
        ] + [
            ("photon", sample) for sample in photon_samples
        ]
        if not self.samples:
            raise ValueError("CombinedCalorimeterDataset requires at least one sample.")
        self.normalization_scale = 0.0 if normalization_scale is None else float(normalization_scale)
        self.phi_window_size = int(phi_window_size)
        self.eta_window_size = int(eta_window_size)
        electron_list = [sample for sample in electron_samples]
        photon_list = [sample for sample in photon_samples]
        self.condition_range = (
            condition_range
            if condition_range is not None
            else condition_bounds([electron_list, photon_list] if electron_list and photon_list else [electron_list or photon_list])
        )
        self.symmetric_range = bool(symmetric_range)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        domain_name, sample = self.samples[index]
        return {
            "image": normalize_window(
                sample.image,
                self.normalization_scale,
                self.phi_window_size,
                self.eta_window_size,
                symmetric_range=self.symmetric_range,
            ),
            "cond": torch.from_numpy(normalize_condition(sample, self.condition_range)),
            "energy_scale": torch.tensor([particle_energy_scale_mev(sample)], dtype=torch.float32),
            "path": sample_identifier(sample),
            "domain": domain_name,
            "domain_index": torch.tensor(domain_to_index(domain_name), dtype=torch.long),
        }
