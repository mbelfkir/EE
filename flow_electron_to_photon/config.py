from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from flow_electron_to_photon.processor.config import Step


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RootDomainSpec(StrictModel):
    file_paths: list[str]
    tree_name: str
    branch_overrides: Dict[str, str | None] = Field(default_factory=dict)


class IOSpec(StrictModel):
    source: RootDomainSpec
    target: RootDomainSpec
    source_label: str = "electrons"
    target_label: str = "photons"


class BranchSpec(StrictModel):
    cell_energy: str
    cell_x: str
    cell_y: str
    cell_layer: str
    cell_dx: str | None = None
    cell_dy: str | None = None
    layer1_energy: str | None = None
    layer2_energy: str | None = None
    layer3_energy: str | None = None
    cluster_energy: str | None = None
    total_energy: str | None = None
    particle_energy: str | None = None
    particle_px: str | None = None
    particle_py: str | None = None
    particle_pz: str | None = None
    particle_pdgid: str | None = None
    event_weight: str | None = None
    event_number: str | None = None


class PhysicsFeatureSpec(StrictModel):
    eta_phi_unit_mm: float = 1440.0
    default_center_layer: int = 2
    default_layer_cell_sizes: Dict[str, tuple[float, float]] = Field(default_factory=dict)
    feature_builders: Dict[str, Dict[str, Any]]


class RootSpec(StrictModel):
    branches: BranchSpec
    physics: PhysicsFeatureSpec


class RuntimeSpec(StrictModel):
    output_path: str
    seed: int = 1
    copy_config: bool = True


class ProcessorSpec(StrictModel):
    pre: list[Step]
    post: list[Step]


class AppConfig(StrictModel):
    runtime: RuntimeSpec
    io: IOSpec
    root: RootSpec
    processor: ProcessorSpec

    @classmethod
    def load_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)
