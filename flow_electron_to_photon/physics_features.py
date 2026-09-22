from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import awkward as ak
import numpy as np

from flow_electron_to_photon.config import BranchSpec, PhysicsFeatureSpec


def _as_float_scalar(value: Any) -> float:
    if isinstance(value, ak.Array):
        array = ak.to_numpy(value)
        if array.ndim == 0:
            return float(array)
        if array.size == 0:
            return 0.0
        return float(array.reshape(-1)[0])
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        if value.size == 0:
            return 0.0
        return float(value.reshape(-1)[0])
    return float(value)


def _safe_eta(px: float, py: float, pz: float) -> float:
    momentum = math.sqrt(px * px + py * py + pz * pz)
    denominator = max(momentum - pz, 1e-12)
    numerator = max(momentum + pz, 1e-12)
    return 0.5 * math.log(numerator / denominator)


@dataclass(frozen=True)
class CellCenter:
    eta: float
    phi: float


class FeatureBuildError(RuntimeError):
    pass


class EventFeatureComputer:
    def __init__(
        self,
        arrays: Mapping[str, ak.Array],
        index: int,
        branches: BranchSpec,
        physics: PhysicsFeatureSpec,
        *,
        file_path: str,
        entry_index: int,
        domain_name: str,
        branch_overrides: Mapping[str, str | None] | None = None,
    ) -> None:
        self.arrays = arrays
        self.index = int(index)
        self.branches = branches
        self.physics = physics
        self.file_path = file_path
        self.entry_index = int(entry_index)
        self.domain_name = domain_name
        self.branch_overrides = dict(branch_overrides or {})
        self.feature_cache: dict[str, float] = {}
        self.center_cache: dict[int, CellCenter] = {}
        self.profile_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.window_cache: dict[tuple[int, int, int, int, int], float] = {}
        self.width_cache: dict[tuple[int, int, int, int, int, str], float] = {}
        self._cell_cache: dict[str, np.ndarray] | None = None

    def compute(self, name: str, spec: Mapping[str, Any]) -> float:
        if name in self.feature_cache:
            return self.feature_cache[name]

        feature_type = str(spec.get("type", "")).strip().lower()
        if feature_type == "":
            raise FeatureBuildError(f"Feature '{name}' is missing a 'type'.")

        if feature_type in {"branch", "copy_branch"}:
            value = self.scalar_branch(str(spec["branch"]))
        elif feature_type == "constant":
            value = float(spec["value"])
        elif feature_type == "entry_index":
            value = float(self.entry_index)
        elif feature_type == "momentum_pt":
            px = self.scalar_alias("particle_px")
            py = self.scalar_alias("particle_py")
            value = math.sqrt(px * px + py * py)
        elif feature_type == "momentum_eta":
            px = self.scalar_alias("particle_px")
            py = self.scalar_alias("particle_py")
            pz = self.scalar_alias("particle_pz")
            value = _safe_eta(px, py, pz)
        elif feature_type == "momentum_phi":
            px = self.scalar_alias("particle_px")
            py = self.scalar_alias("particle_py")
            value = math.atan2(py, px)
        elif feature_type == "energy_fraction":
            numerator = self.scalar_branch(str(spec["numerator"]))
            denominator = self.scalar_branch(str(spec["denominator"]))
            value = 0.0 if denominator == 0.0 else numerator / denominator
        elif feature_type == "layer_window_sum":
            value = self.layer_window_sum(
                layer=int(spec["layer"]),
                eta_bins=int(spec["eta_bins"]),
                phi_bins=int(spec["phi_bins"]),
                center_layer=int(spec.get("center_layer", spec["layer"])),
            )
        elif feature_type == "layer_ratio":
            numerator = spec["numerator_window"]
            denominator = spec["denominator_window"]
            numerator_sum = self.layer_window_sum(
                layer=int(spec["layer"]),
                eta_bins=int(numerator[0]),
                phi_bins=int(numerator[1]),
                center_layer=int(spec.get("center_layer", spec["layer"])),
            )
            denominator_sum = self.layer_window_sum(
                layer=int(spec["layer"]),
                eta_bins=int(denominator[0]),
                phi_bins=int(denominator[1]),
                center_layer=int(spec.get("center_layer", spec["layer"])),
            )
            value = 0.0 if denominator_sum == 0.0 else numerator_sum / denominator_sum
        elif feature_type == "layer_width":
            value = self.layer_window_width(
                layer=int(spec["layer"]),
                eta_bins=int(spec["eta_bins"]),
                phi_bins=int(spec["phi_bins"]),
                center_layer=int(spec.get("center_layer", spec["layer"])),
                axis=str(spec.get("axis", "eta")),
            )
        elif feature_type == "side_fraction":
            value = self.side_fraction(
                layer=int(spec["layer"]),
                outer_eta_bins=int(spec["outer_eta_bins"]),
                core_eta_bins=int(spec["core_eta_bins"]),
                phi_bins=int(spec["phi_bins"]),
                center_layer=int(spec.get("center_layer", spec["layer"])),
                denominator=str(spec.get("denominator", "outer")),
            )
        elif feature_type == "strip_deltae":
            value = self.strip_deltae(
                layer=int(spec.get("layer", 1)),
                eta_bins=int(spec.get("eta_bins", 21)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", spec.get("layer", 1))),
                min_peak_gap_bins=int(spec.get("min_peak_gap_bins", 2)),
            )
        elif feature_type == "strip_eratio":
            value = self.strip_eratio(
                layer=int(spec.get("layer", 1)),
                eta_bins=int(spec.get("eta_bins", 21)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", spec.get("layer", 1))),
                min_peak_gap_bins=int(spec.get("min_peak_gap_bins", 2)),
            )
        elif feature_type == "layer2_reta":
            numerator = self.layer_window_sum(layer=2, eta_bins=3, phi_bins=7, center_layer=int(spec.get("center_layer", 2)))
            denominator = self.layer_window_sum(layer=2, eta_bins=7, phi_bins=7, center_layer=int(spec.get("center_layer", 2)))
            value = 0.0 if denominator == 0.0 else numerator / denominator
        elif feature_type == "layer2_rphi":
            numerator = self.layer_window_sum(layer=2, eta_bins=3, phi_bins=3, center_layer=int(spec.get("center_layer", 2)))
            denominator = self.layer_window_sum(layer=2, eta_bins=3, phi_bins=7, center_layer=int(spec.get("center_layer", 2)))
            value = 0.0 if denominator == 0.0 else numerator / denominator
        elif feature_type == "layer2_weta2":
            value = self.layer_window_width(
                layer=2,
                eta_bins=int(spec.get("eta_bins", 3)),
                phi_bins=int(spec.get("phi_bins", 5)),
                center_layer=int(spec.get("center_layer", 2)),
                axis="eta",
            )
        elif feature_type == "layer2_wphi2":
            value = self.layer_window_width(
                layer=2,
                eta_bins=int(spec.get("eta_bins", 3)),
                phi_bins=int(spec.get("phi_bins", 5)),
                center_layer=int(spec.get("center_layer", 2)),
                axis="phi",
            )
        elif feature_type == "layer1_wstot":
            value = self.layer_window_width(
                layer=1,
                eta_bins=int(spec.get("eta_bins", 20)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", 1)),
                axis="eta",
            )
        elif feature_type == "layer1_fside":
            value = self.side_fraction(
                layer=1,
                outer_eta_bins=int(spec.get("outer_eta_bins", 7)),
                core_eta_bins=int(spec.get("core_eta_bins", 3)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", 1)),
                denominator=str(spec.get("denominator", "outer")),
            )
        elif feature_type == "layer1_deltae":
            value = self.strip_deltae(
                layer=1,
                eta_bins=int(spec.get("eta_bins", 21)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", 1)),
                min_peak_gap_bins=int(spec.get("min_peak_gap_bins", 2)),
            )
        elif feature_type == "layer1_eratio":
            value = self.strip_eratio(
                layer=1,
                eta_bins=int(spec.get("eta_bins", 21)),
                phi_bins=int(spec.get("phi_bins", 3)),
                center_layer=int(spec.get("center_layer", 1)),
                min_peak_gap_bins=int(spec.get("min_peak_gap_bins", 2)),
            )
        else:
            raise FeatureBuildError(f"Unsupported feature builder '{feature_type}' for '{name}'.")

        if not np.isfinite(value):
            raise FeatureBuildError(
                f"Feature '{name}' evaluated to a non-finite value in {Path(self.file_path).name} "
                f"entry {self.entry_index}."
            )

        self.feature_cache[name] = float(value)
        return float(value)

    def _branch_name(self, alias_or_name: str) -> str:
        if alias_or_name in self.branch_overrides:
            override = self.branch_overrides[alias_or_name]
            if override is None:
                raise FeatureBuildError(f"Branch alias '{alias_or_name}' is explicitly disabled for this domain.")
            return override
        if hasattr(self.branches, alias_or_name):
            value = getattr(self.branches, alias_or_name)
            if value is None:
                raise FeatureBuildError(f"Branch alias '{alias_or_name}' is not configured.")
            return value
        return alias_or_name

    def scalar_alias(self, alias_name: str) -> float:
        return self.scalar_branch(alias_name)

    def scalar_branch(self, alias_or_name: str) -> float:
        branch_name = self._branch_name(alias_or_name)
        if branch_name not in self.arrays:
            raise FeatureBuildError(f"Branch '{branch_name}' is not loaded for feature computation.")
        return _as_float_scalar(self.arrays[branch_name][self.index])

    @property
    def cells(self) -> dict[str, np.ndarray]:
        if self._cell_cache is not None:
            return self._cell_cache

        energy = ak.to_numpy(self.arrays[self._branch_name("cell_energy")][self.index]).astype(np.float64)
        x = ak.to_numpy(self.arrays[self._branch_name("cell_x")][self.index]).astype(np.float64)
        y = ak.to_numpy(self.arrays[self._branch_name("cell_y")][self.index]).astype(np.float64)
        layer = ak.to_numpy(self.arrays[self._branch_name("cell_layer")][self.index]).astype(np.int64)
        eta = x / float(self.physics.eta_phi_unit_mm)
        phi = y / float(self.physics.eta_phi_unit_mm)

        if self.branches.cell_dx is not None:
            dx = ak.to_numpy(self.arrays[self._branch_name("cell_dx")][self.index]).astype(np.float64)
            eta_size = dx / float(self.physics.eta_phi_unit_mm)
        else:
            eta_size = np.zeros_like(eta)

        if self.branches.cell_dy is not None:
            dy = ak.to_numpy(self.arrays[self._branch_name("cell_dy")][self.index]).astype(np.float64)
            phi_size = dy / float(self.physics.eta_phi_unit_mm)
        else:
            phi_size = np.zeros_like(phi)

        self._cell_cache = {
            "energy": energy,
            "eta": eta,
            "phi": phi,
            "layer": layer,
            "eta_size": eta_size,
            "phi_size": phi_size,
        }
        return self._cell_cache

    def layer_cell_size(self, layer: int) -> tuple[float, float]:
        configured = self.physics.default_layer_cell_sizes.get(str(layer))
        if configured is not None:
            return float(configured[0]), float(configured[1])

        mask = self.cells["layer"] == int(layer)
        if mask.any():
            eta_sizes = self.cells["eta_size"][mask]
            phi_sizes = self.cells["phi_size"][mask]
            eta_size = float(np.median(eta_sizes[eta_sizes > 0])) if np.any(eta_sizes > 0) else 0.025
            phi_size = float(np.median(phi_sizes[phi_sizes > 0])) if np.any(phi_sizes > 0) else 0.025
            return eta_size, phi_size

        return 0.025, 0.025

    def hot_cell_center(self, layer: int) -> CellCenter:
        if layer in self.center_cache:
            return self.center_cache[layer]

        mask = self.cells["layer"] == int(layer)
        if not mask.any():
            raise FeatureBuildError(
                f"No cells found in layer {layer} for {Path(self.file_path).name} entry {self.entry_index}."
            )

        energies = self.cells["energy"][mask]
        local_index = int(np.argmax(energies))
        eta = float(self.cells["eta"][mask][local_index])
        phi = float(self.cells["phi"][mask][local_index])
        center = CellCenter(eta=eta, phi=phi)
        self.center_cache[layer] = center
        return center

    def layer_window_sum(self, layer: int, eta_bins: int, phi_bins: int, center_layer: int) -> float:
        key = (int(layer), int(eta_bins), int(phi_bins), int(center_layer), 0)
        if key in self.window_cache:
            return self.window_cache[key]

        mask = self.cells["layer"] == int(layer)
        if not mask.any():
            self.window_cache[key] = 0.0
            return 0.0

        center = self.hot_cell_center(int(center_layer))
        eta_size, phi_size = self.layer_cell_size(int(layer))
        eta_half = eta_bins // 2
        phi_half = phi_bins // 2

        deta_bins = np.rint((self.cells["eta"][mask] - center.eta) / eta_size).astype(np.int64)
        dphi_bins = np.rint((self.cells["phi"][mask] - center.phi) / phi_size).astype(np.int64)
        select = (np.abs(deta_bins) <= eta_half) & (np.abs(dphi_bins) <= phi_half)
        value = float(self.cells["energy"][mask][select].sum())
        self.window_cache[key] = value
        return value

    def layer_window_width(
        self,
        layer: int,
        eta_bins: int,
        phi_bins: int,
        center_layer: int,
        axis: str,
    ) -> float:
        axis_name = str(axis).lower()
        key = (int(layer), int(eta_bins), int(phi_bins), int(center_layer), axis_name)
        if key in self.width_cache:
            return self.width_cache[key]

        mask = self.cells["layer"] == int(layer)
        if not mask.any():
            self.width_cache[key] = 0.0
            return 0.0

        center = self.hot_cell_center(int(center_layer))
        eta_size, phi_size = self.layer_cell_size(int(layer))
        eta_half = eta_bins // 2
        phi_half = phi_bins // 2
        deta = self.cells["eta"][mask] - center.eta
        dphi = self.cells["phi"][mask] - center.phi
        deta_bins = np.rint(deta / eta_size).astype(np.int64)
        dphi_bins = np.rint(dphi / phi_size).astype(np.int64)
        select = (np.abs(deta_bins) <= eta_half) & (np.abs(dphi_bins) <= phi_half)
        energies = self.cells["energy"][mask][select]
        coordinates = deta[select] if axis_name == "eta" else dphi[select]
        total = float(energies.sum())
        if total <= 0.0:
            self.width_cache[key] = 0.0
            return 0.0
        mean = float((energies * coordinates).sum() / total)
        mean2 = float((energies * (coordinates ** 2)).sum() / total)
        variance = max(0.0, mean2 - mean * mean)
        value = math.sqrt(variance)
        self.width_cache[key] = value
        return value

    def eta_profile(self, layer: int, eta_bins: int, phi_bins: int, center_layer: int) -> np.ndarray:
        key = (int(layer), int(eta_bins), int(phi_bins), int(center_layer))
        if key in self.profile_cache:
            return self.profile_cache[key]

        mask = self.cells["layer"] == int(layer)
        if not mask.any():
            self.profile_cache[key] = np.zeros(int(eta_bins), dtype=np.float64)
            return self.profile_cache[key]

        center = self.hot_cell_center(int(center_layer))
        eta_size, phi_size = self.layer_cell_size(int(layer))
        eta_half = eta_bins // 2
        phi_half = phi_bins // 2
        deta_bins = np.rint((self.cells["eta"][mask] - center.eta) / eta_size).astype(np.int64)
        dphi_bins = np.rint((self.cells["phi"][mask] - center.phi) / phi_size).astype(np.int64)

        profile = np.zeros(int(eta_bins), dtype=np.float64)
        for energy, eta_bin, phi_bin in zip(self.cells["energy"][mask], deta_bins, dphi_bins):
            if abs(phi_bin) > phi_half or abs(eta_bin) > eta_half:
                continue
            profile[int(eta_bin) + eta_half] += float(energy)

        self.profile_cache[key] = profile
        return profile

    def side_fraction(
        self,
        layer: int,
        outer_eta_bins: int,
        core_eta_bins: int,
        phi_bins: int,
        center_layer: int,
        denominator: str,
    ) -> float:
        profile = self.eta_profile(
            layer=int(layer),
            eta_bins=int(outer_eta_bins),
            phi_bins=int(phi_bins),
            center_layer=int(center_layer),
        )
        if profile.size == 0:
            return 0.0
        center = profile.size // 2
        core_half = core_eta_bins // 2
        core_sum = float(profile[center - core_half:center + core_half + 1].sum())
        outer_sum = float(profile.sum())
        side_sum = max(0.0, outer_sum - core_sum)
        denom_key = str(denominator).lower()
        if denom_key == "core":
            denom = core_sum
        else:
            denom = outer_sum
        return 0.0 if denom <= 0.0 else side_sum / denom

    def _strip_peak_summary(
        self,
        layer: int,
        eta_bins: int,
        phi_bins: int,
        center_layer: int,
        min_peak_gap_bins: int,
    ) -> tuple[float, float, float]:
        profile = self.eta_profile(
            layer=int(layer),
            eta_bins=int(eta_bins),
            phi_bins=int(phi_bins),
            center_layer=int(center_layer),
        )
        if profile.size == 0:
            return 0.0, 0.0, 0.0

        primary_index = int(np.argmax(profile))
        primary_max = float(profile[primary_index])
        secondary_profile = profile.copy()
        secondary_profile[max(0, primary_index - min_peak_gap_bins):primary_index + min_peak_gap_bins + 1] = -np.inf
        if not np.isfinite(secondary_profile).any():
            return primary_max, 0.0, 0.0

        secondary_index = int(np.argmax(secondary_profile))
        secondary_max = max(float(profile[secondary_index]), 0.0)
        lo = min(primary_index, secondary_index)
        hi = max(primary_index, secondary_index)
        valley = 0.0 if hi <= lo else float(profile[lo:hi + 1].min())
        return primary_max, secondary_max, valley

    def strip_deltae(
        self,
        layer: int,
        eta_bins: int,
        phi_bins: int,
        center_layer: int,
        min_peak_gap_bins: int,
    ) -> float:
        _, secondary_max, valley = self._strip_peak_summary(
            layer=int(layer),
            eta_bins=int(eta_bins),
            phi_bins=int(phi_bins),
            center_layer=int(center_layer),
            min_peak_gap_bins=int(min_peak_gap_bins),
        )
        return max(0.0, secondary_max - valley)

    def strip_eratio(
        self,
        layer: int,
        eta_bins: int,
        phi_bins: int,
        center_layer: int,
        min_peak_gap_bins: int,
    ) -> float:
        primary_max, secondary_max, _ = self._strip_peak_summary(
            layer=int(layer),
            eta_bins=int(eta_bins),
            phi_bins=int(phi_bins),
            center_layer=int(center_layer),
            min_peak_gap_bins=int(min_peak_gap_bins),
        )
        denom = primary_max + secondary_max
        if denom <= 0.0:
            return 0.0
        return (primary_max - secondary_max) / denom
