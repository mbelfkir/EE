from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    name: str | None = Field(default=None, validate_default=True)
    enabled: bool = True

    @field_validator("name", mode="before")
    @classmethod
    def fill_name(cls, value: str | None, info):
        if value is None:
            return info.data["type"]
        return value


class NoWeightsStep(StrictModel):
    type: Literal["no_weights"]
    weight_var: str


class QueryCutsStep(StrictModel):
    type: Literal["query_cuts"]
    expressions: List[str]


class StandardizeStep(StrictModel):
    type: Literal["standardize"]
    weighted: bool
    weight_var: str
    variables: list[str]
    post_mapping: Dict[str, str]
    mode: Literal["fit", "load"] = "fit"
    stats_path: str | None = None

    @model_validator(mode="after")
    def validate_mapping(self) -> "StandardizeStep":
        diff = set(self.post_mapping.values()) - set(self.variables)
        if diff:
            raise ValueError(f"Unknown variables in post_mapping: {list(diff)}")
        return self


class TruncateEventsStep(StrictModel):
    type: Literal["truncate"]
    offset: int = 0
    limit: int


class CastToFloatStep(StrictModel):
    type: Literal["cast_to_float"]
    variables: List[str]


class SmoothStep(StrictModel):
    type: Literal["smooth"]
    eratio_variables: List[str]
    deltae_variables: List[str]


class PrinterStep(StrictModel):
    type: Literal["printer"]
    name: str | None = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""


class PlotVariablesStep(StrictModel):
    type: Literal["plot_variables"]
    variables: List[str]
    weight_var: str
    variable_labels: Dict[str, str]
    binning: Dict[str, Tuple[int, float, float]]
    annotation: str
    subannotation: str
    ylabel: str
    ratio_ylabel: str
    legend_labels: Dict[str, str]


class SplitTrainTestValidationStep(StrictModel):
    type: Literal["split_train_test_validation"]
    nfolds: int
    fold_seed_variable: Optional[str] = None
    train_to_validation_ratio: float
    split_mode: Literal["folds", "holdout"] = "folds"
    test_fraction: float | None = None
    stratify_by_type: bool = True

    @model_validator(mode="after")
    def validate_split_mode(self) -> "SplitTrainTestValidationStep":
        if self.split_mode == "holdout":
            if self.nfolds != 1:
                raise ValueError("holdout split_mode requires nfolds=1")
            if self.test_fraction is None:
                raise ValueError("holdout split_mode requires test_fraction")
            if not 0.0 < self.test_fraction < 1.0:
                raise ValueError("test_fraction must be between 0 and 1")
        elif self.nfolds < 2:
            raise ValueError("folds split_mode requires nfolds>=2")
        return self


class TrainApplyFlowStep(StrictModel):
    type: Literal["train_apply_flow"]
    kinematic: List[str]
    shower_shapes: List[str]
    weight_var: str
    n_transforms: int
    max_epoch: int
    points_per_record: int
    points_per_epoch: int
    aux_nodes: int
    aux_layers: int
    n_splines_bins: int
    initial_lr: float
    batch_size: int
    weighted: bool
    nfolds: int
    corr_suffix: str = "_NF"
    fudged_prefix: str = ""
    variable_labels: Dict[str, str]
    binning: Dict[str, Tuple[int, float, float]]
    annotation: str
    subannotation: str
    ylabel: str
    ratio_ylabel: str
    legend_labels: Dict[str, str]
    kinematic_slices: Dict[str, List[float]] = Field(default_factory=dict)
    smooth_step_name: str | None = None
    standardize_step_name: str | None = None


class ShuffleStep(StrictModel):
    type: Literal["dataset_shuffle"]


class PlotCorrectedVariablesStep(StrictModel):
    type: Literal["plot_corrected_variables"]
    variables: List[str]
    weight_var: str
    variable_labels: Dict[str, str]
    binning: Dict[str, Tuple[int, float, float]]
    annotation: str
    subannotation: str
    ylabel: str
    ratio_ylabel: str
    legend_labels: Dict[str, str]
    corr_suffix: str = "_NF"


class PlotCorrectedFudgedVariablesStep(StrictModel):
    type: Literal["plot_corrected_fudged_variables"]
    variables: List[str]
    weight_var: str
    variable_labels: Dict[str, str]
    binning: Dict[str, Tuple[int, float, float]]
    annotation: str
    subannotation: str
    ylabel: str
    ratio_ylabel: str
    legend_labels: Dict[str, str]
    corr_suffix: str = "_NF"
    fudged_prefix: str


class ExportSourceStep(StrictModel):
    type: Literal["export_source"]
    filename: str
    format: Literal["hdf5", "parquet", "root"] = "hdf5"


class PlotCorrelationMatricesStep(StrictModel):
    type: Literal["plot_correlation_matrices"]
    variables: List[str]
    weight_var: str
    variable_labels: Dict[str, str]
    corr_suffix: str = "_NF"


class PlotCorrectedVariablesSlicesStep(StrictModel):
    type: Literal["plot_corrected_variables_slices"]
    variables: List[str]
    weight_var: str
    variable_labels: Dict[str, str]
    binning: Dict[str, Tuple[int, float, float]]
    annotation: str
    subannotation: str
    ylabel: str
    ratio_ylabel: str
    legend_labels: Dict[str, str]
    corr_suffix: str = "_NF"
    kinematic_slices: Dict[str, List[float]]


class ApplyFoldMarkupStep(StrictModel):
    type: Literal["apply_fold_markup"]
    nfolds: int
    fold_seed_variable: Optional[str] = None
    column: str = "fold_id"
    apply_to_target: bool = False


class ApplySavedFlowFoldsStep(StrictModel):
    type: Literal["apply_saved_flow_folds"]
    nfolds: int
    model_paths: List[str]
    kinematic: List[str]
    shower_shapes: List[str]
    weight_var: str
    corr_suffix: str = "_NF"
    fold_column: str = "fold_id"
    batch_size: int = 8192

    @model_validator(mode="after")
    def validate_paths(self) -> "ApplySavedFlowFoldsStep":
        if len(self.model_paths) != self.nfolds:
            raise ValueError("model_paths length must match nfolds")
        return self


Step = Annotated[
    Union[
        NoWeightsStep,
        QueryCutsStep,
        StandardizeStep,
        TruncateEventsStep,
        CastToFloatStep,
        SmoothStep,
        PrinterStep,
        PlotVariablesStep,
        SplitTrainTestValidationStep,
        TrainApplyFlowStep,
        ShuffleStep,
        PlotCorrectedVariablesStep,
        PlotCorrectedFudgedVariablesStep,
        ExportSourceStep,
        PlotCorrelationMatricesStep,
        PlotCorrectedVariablesSlicesStep,
        ApplyFoldMarkupStep,
        ApplySavedFlowFoldsStep,
    ],
    Field(discriminator="type"),
]
