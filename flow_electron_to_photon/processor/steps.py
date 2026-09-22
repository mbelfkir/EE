from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from flow_electron_to_photon.misc.system import export_numeric_dataframe, load_stats_hdf5, save_stats_hdf5, setup_output_dir
from flow_electron_to_photon.misc.train import (
    TEST_LABEL,
    TRAIN_LABEL,
    VALIDATION_LABEL,
    Applier,
    SmoothingConfig,
    Trainer,
    load_flow_from_checkpoint,
)
from flow_electron_to_photon.plotting.hists import (
    Binning,
    plot_corrected_fudged_hists,
    plot_corrected_hists,
    plot_input_hists,
)
from flow_electron_to_photon.processor.plotting import plot_corrected_hists_slices, plot_correlation_matrices
from flow_electron_to_photon.root_loader import extract_columns_from_query


RANDOM_STATE = 1


class NoOpStep:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return proto, target

    @property
    def state(self) -> dict[str, Any]:
        return {}

    @state.setter
    def state(self, state: dict[str, Any]) -> None:
        return

    @property
    def used_variables(self) -> Iterable[str]:
        return []


class Standardize(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: list[str],
        post_mapping: Dict[str, str],
        weight_var: str,
        outpath: str,
        *,
        enabled: bool = True,
        weighted: bool = True,
        mode: str = "fit",
        stats_path: str | None = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variables = variables
        self.post_mapping = post_mapping
        self.weight_var = weight_var
        self.weighted = weighted
        self.outpath = outpath
        self.mode = mode
        self.stats_path = stats_path
        self.mean: Optional[pd.Series] = None
        self.std: Optional[pd.Series] = None

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + [self.weight_var]

    @property
    def resolved_stats_path(self) -> str:
        return self.stats_path or os.path.join(self.outpath, "MeanAndStd.h5")

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target

        if self.mode == "fit":
            self.prepare(proto)
        elif self.mean is None or self.std is None:
            self.load(self.resolved_stats_path)

        return self.forward(proto), self.forward(target)

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        return self.backward(proto), self.backward(target)

    def prepare(self, df: pd.DataFrame) -> None:
        if self.weighted:
            w = df[self.weight_var].to_numpy(dtype=np.float64)
        else:
            w = np.ones(len(df), dtype=np.float64)

        values = df[self.variables].to_numpy(dtype=np.float64)
        mean = np.average(values, weights=w, axis=0)
        var = np.average((values - mean) ** 2, weights=w, axis=0)
        std = np.sqrt(var)
        std = np.where(std > 0.0, std, 1.0)

        self.mean = pd.Series(mean, index=self.variables, dtype=np.float64)
        self.std = pd.Series(std, index=self.variables, dtype=np.float64)
        save_stats_hdf5(self.mean, self.std, self.resolved_stats_path)
        print(f"\033[1;36m[INFO]\033[92m Mean/std saved to \033[0m {self.resolved_stats_path}")

    def load(self, path: str | Path) -> None:
        self.mean, self.std = load_stats_hdf5(path)
        missing = [var for var in self.variables if var not in self.mean.index or var not in self.std.index]
        if missing:
            raise KeyError(f"Standardization statistics are missing variables: {missing}")
        self.mean = self.mean[self.variables]
        self.std = self.std[self.variables]

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardize not prepared")
        df.loc[:, self.variables] = ((df.loc[:, self.variables] - self.mean) / self.std).astype(df[self.variables].dtypes)
        return df

    def backward(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardize not prepared")

        additional_columns = list(set(df.columns) & set(self.post_mapping))
        mapped_columns = [self.post_mapping[column] for column in additional_columns]

        df.loc[:, self.variables] = (df.loc[:, self.variables] * self.std + self.mean).astype(df[self.variables].dtypes)
        if additional_columns:
            df.loc[:, additional_columns] = (
                df.loc[:, additional_columns] * self.std[mapped_columns].to_numpy() + self.mean[mapped_columns].to_numpy()
            ).astype(df[additional_columns].dtypes)
        return df

    @property
    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "variables": self.variables,
            "mean": None if self.mean is None else self.mean.to_dict(),
            "std": None if self.std is None else self.std.to_dict(),
        }

    @state.setter
    def state(self, state: dict[str, Any]) -> None:
        self.enabled = state["enabled"]
        self.variables = state["variables"]
        self.mean = pd.Series(state["mean"])
        self.std = pd.Series(state["std"])


class NoWeights(NoOpStep):
    def __init__(self, name: str, weight_var: str, enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.weight_var = weight_var

    @property
    def used_variables(self) -> Iterable[str]:
        return [self.weight_var]

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        proto[self.weight_var] = 1.0
        target[self.weight_var] = 1.0
        return proto, target


class QueryCuts(NoOpStep):
    def __init__(self, name: str, expressions: list[str], enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.expressions = expressions

    @property
    def used_variables(self) -> Iterable[str]:
        variables = set()
        for expr in self.expressions:
            variables |= extract_columns_from_query(expr)
        return variables

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        return self.transform(proto), self.transform(target)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.expressions:
            return df
        mapping = {column: column.replace(".", "_") for column in df.columns}
        df = df.rename(columns=mapping)
        for expr in self.expressions:
            translated = expr
            for old, new in mapping.items():
                translated = translated.replace(old, new)
            try:
                df = df.query(translated, engine="numexpr")
            except Exception:
                df = df.query(translated, engine="python")
        return df.rename(columns={new: old for old, new in mapping.items()})


class TruncateEvents(NoOpStep):
    def __init__(self, name: str, offset: int, limit: int, enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.offset = offset
        self.limit = limit

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        return proto.iloc[self.offset:self.offset + self.limit], target.iloc[self.offset:self.offset + self.limit]


class CastToFloat(NoOpStep):
    def __init__(self, name: str, variables: Iterable[str], enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.variables = list(variables)

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        proto[self.variables] = proto[self.variables].astype(float)
        target[self.variables] = target[self.variables].astype(float)
        return proto, target


class Smooth(NoOpStep):
    def __init__(
        self,
        name: str,
        eratio_variables: Iterable[str],
        deltae_variables: Iterable[str],
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.eratio_variables = list(eratio_variables)
        self.deltae_variables = list(deltae_variables)

    @property
    def used_variables(self) -> Iterable[str]:
        variables = [*self.eratio_variables, *self.deltae_variables]
        return [variable for variable in variables if not variable.endswith("_NF")]

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        return self.forward(proto), self.forward(target)

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        return self.backward(proto), self.backward(target)

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.eratio_variables:
            if column in df.columns:
                self.smooth_eratio(df, column)
        for column in self.deltae_variables:
            if column in df.columns:
                self.smooth_deltae(df, column)
        return df

    def backward(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.eratio_variables:
            if column in df.columns:
                self.desmooth_eratio(df, column)
        for column in self.deltae_variables:
            if column in df.columns:
                self.desmooth_deltae(df, column)
        return df

    def smooth_eratio(self, df: pd.DataFrame, col: str, shift: float = 0.1, eps: float = 1e-3) -> None:
        left = 1.0 + shift
        mask = df[col] >= 1.0
        if mask.any():
            df.loc[mask, col] = np.random.triangular(left=left, mode=left, right=left + shift, size=int(mask.sum())).astype(df[col].dtype)
        df[col] = np.log(df[col] + eps)

    def smooth_deltae(self, df: pd.DataFrame, col: str, eps: float = 1e-3) -> None:
        mask = df[col] == 0.0
        df.loc[~mask, col] = df.loc[~mask, col] + 3.0
        if mask.any():
            df.loc[mask, col] = np.random.triangular(left=1.0, mode=1.0, right=2.0, size=int(mask.sum())).astype(df[col].dtype)
        df[col] = np.log(df[col] + eps)

    def desmooth_eratio(self, df: pd.DataFrame, col: str, eps: float = 1e-3) -> None:
        df[col] = np.exp(df[col]) - eps
        mask = df[col] > 1.0
        if mask.any():
            df.loc[mask, col] = 1.0

    def desmooth_deltae(self, df: pd.DataFrame, col: str, eps: float = 1e-3) -> None:
        df[col] = np.exp(df[col]) - eps
        mask = df[col] < 3.0
        if mask.any():
            df.loc[mask, col] = 0.0
        if (~mask).any():
            df.loc[~mask, col] = df.loc[~mask, col] - 3.0


class ApplyFoldMarkup(NoOpStep):
    def __init__(
        self,
        name: str,
        nfolds: int,
        fold_seed_variable: Optional[str],
        column: str,
        apply_to_target: bool,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.nfolds = nfolds
        self.fold_seed_variable = fold_seed_variable
        self.column = column
        self.apply_to_target = apply_to_target

    @property
    def used_variables(self) -> Iterable[str]:
        return [self.fold_seed_variable] if self.fold_seed_variable is not None else []

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        self.markup(proto, dataset="source")
        if self.apply_to_target:
            self.markup(target, dataset="target")
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("not supported in post context")

    def markup(self, df: pd.DataFrame, dataset: str) -> None:
        if df.empty:
            df[self.column] = pd.Series(dtype=np.int32)
            return

        if self.fold_seed_variable is not None:
            if self.fold_seed_variable not in df.columns:
                raise KeyError(f"{self.fold_seed_variable} not found in {dataset} dataset for fold markup")
            seeds = df[self.fold_seed_variable].astype(np.int64, copy=False).to_numpy()
            fold_ids = (seeds % self.nfolds).astype(np.int32, copy=False)
        else:
            idx = np.arange(len(df))
            fold_ids = np.empty(len(df), dtype=np.int32)
            kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=RANDOM_STATE)
            for fold_id, (_, test_idx) in enumerate(kf.split(idx)):
                fold_ids[test_idx] = fold_id
        df[self.column] = fold_ids


class ApplySavedFlowFolds(NoOpStep):
    def __init__(
        self,
        name: str,
        nfolds: int,
        model_paths: List[str],
        kinematic: List[str],
        shower_shapes: List[str],
        weight_var: str,
        corr_suffix: str,
        fold_column: str,
        batch_size: int,
        outpath: str,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        if len(model_paths) != nfolds:
            raise ValueError("model_paths length must match nfolds")
        self.nfolds = nfolds
        self.model_paths = model_paths
        self.kinematic = kinematic
        self.shower_shapes = shower_shapes
        self.weight_var = weight_var
        self.corr_suffix = corr_suffix
        self.fold_column = fold_column
        self.batch_size = batch_size
        self.outpath = outpath

    @property
    def used_variables(self) -> Iterable[str]:
        return set(self.kinematic) | set(self.shower_shapes) | {self.weight_var}

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target

        if self.fold_column not in proto.columns:
            raise KeyError(f"{self.fold_column} column not found in source dataset")

        corr_columns = [f"{var}{self.corr_suffix}" for var in self.shower_shapes]
        for fold_id in range(self.nfolds):
            source_mask = proto[self.fold_column] == fold_id
            if not source_mask.any():
                continue

            fold_outpath = setup_output_dir(os.path.join(self.outpath, f"fold_{fold_id}"), allow_recreate=True)
            flow, metadata = load_flow_from_checkpoint(self.model_paths[fold_id])
            expected_kinematic = self.kinematic + ["type"]
            if metadata.get("kinematic") != expected_kinematic:
                raise ValueError(
                    f"Checkpoint {self.model_paths[fold_id]} expects kinematic {metadata.get('kinematic')}, "
                    f"but config requested {expected_kinematic}."
                )
            if metadata.get("shower_shapes") != self.shower_shapes:
                raise ValueError(
                    f"Checkpoint {self.model_paths[fold_id]} expects shower shapes {metadata.get('shower_shapes')}, "
                    f"but config requested {self.shower_shapes}."
                )

            source_df = proto.loc[source_mask, [*self.kinematic, *self.shower_shapes, self.weight_var]].copy()
            source_df["type"] = 0.0

            if self.fold_column in target.columns:
                target_df = target.loc[target[self.fold_column] == fold_id, [*self.kinematic, *self.shower_shapes, self.weight_var]].copy()
                target_df["type"] = 1.0
            else:
                target_df = pd.DataFrame(columns=[*self.kinematic, *self.shower_shapes, self.weight_var, "type"])

            apply_df = pd.concat([source_df, target_df], axis=0, copy=False)
            applier = Applier(
                flow=flow,
                df=apply_df,
                kinematic=expected_kinematic,
                shower_shapes=self.shower_shapes,
                weight_var=self.weight_var,
                corr_suffix=self.corr_suffix,
                outpath=fold_outpath,
            )
            applier.apply()
            corrected = applier.correct()
            proto.loc[corrected.index, corr_columns] = corrected

        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("not supported in post context")


class Printer(NoOpStep):
    def __init__(self, name: str, text: str, enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.text = text

    def print(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        print(self.text if self.text else self.name)
        print(f"source {'-' * 10}")
        print(proto)
        print()
        print(f"target {'-' * 10}")
        print(target)
        print()

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.print(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.print(proto, target)
        return proto, target


class PlotVariables(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: Iterable[str],
        weight_var: str,
        variable_labels: Dict[str, str],
        binning: Dict[str, Binning],
        annotation: str,
        subannotation: str,
        ylabel: str,
        ratio_ylabel: str,
        legend_labels: Dict[str, str],
        outpath: str,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variables = list(variables)
        self.weight_var = weight_var
        self.binning = binning
        self.variable_labels = variable_labels
        self.annotation = annotation
        self.subannotation = subannotation
        self.ylabel = ylabel
        self.ratio_ylabel = ratio_ylabel
        self.legend_labels = legend_labels
        self.outpath = setup_output_dir(os.path.join(outpath, name), allow_recreate=False)

        for var in self.variables:
            if var not in self.binning:
                raise ValueError(f"variable {var} not found in binning dict")
            if var not in self.variable_labels:
                raise ValueError(f"variable {var} not found in variable_labels dict")

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + [self.weight_var]

    def plot(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        plot_input_hists(
            proto=proto,
            target=target,
            variables=self.variables,
            weight_var=self.weight_var,
            variable_labels=self.variable_labels,
            binning=self.binning,
            annotation=self.annotation,
            subannotation=self.subannotation,
            ylabel=self.ylabel,
            ratio_ylabel=self.ratio_ylabel,
            legend_labels=self.legend_labels,
            outpath=self.outpath,
        )

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target


class SplitTrainTestValidation(NoOpStep):
    def __init__(
        self,
        name: str,
        nfolds: int,
        fold_seed_variable: Optional[str],
        train_to_validation_ratio: float,
        split_mode: str = "folds",
        test_fraction: float | None = None,
        stratify_by_type: bool = True,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.nfolds = nfolds
        self.fold_seed_variable = fold_seed_variable
        self.train_to_validation_ratio = train_to_validation_ratio
        self.split_mode = split_mode
        self.test_fraction = test_fraction
        self.stratify_by_type = stratify_by_type

    @property
    def used_variables(self) -> Iterable[str]:
        return [self.fold_seed_variable] if self.fold_seed_variable is not None else []

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.markup(proto, target)

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("not supported in post context")

    def markup(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        proto["type"], target["type"] = 0, 1
        df = pd.concat([proto, target], axis=0, ignore_index=True, copy=False)

        if self.split_mode == "holdout":
            split = np.full(len(df), TRAIN_LABEL, dtype=np.int8)
            indices = np.arange(len(df))
            stratify = df["type"] if self.stratify_by_type else None
            _, test_idx = train_test_split(
                indices,
                test_size=self.test_fraction,
                random_state=RANDOM_STATE,
                shuffle=True,
                stratify=stratify,
            )
            split[test_idx] = TEST_LABEL
            df["fold_0"] = split
            return df[df["type"] == 0], df[df["type"] == 1]

        if self.fold_seed_variable is not None:
            seed_vals = df[self.fold_seed_variable].astype(np.int64).to_numpy()
            test_fold = seed_vals % self.nfolds
            for fold_id in range(self.nfolds):
                split = np.full(len(df), -1, dtype=np.int8)
                test_mask = test_fold == fold_id
                test_idx = np.flatnonzero(test_mask)
                train_validation_idx = np.flatnonzero(~test_mask)
                split[test_idx] = TEST_LABEL
                train_idx, validation_idx = train_test_split(
                    train_validation_idx,
                    test_size=1.0 / (self.train_to_validation_ratio + 1),
                    random_state=RANDOM_STATE,
                    shuffle=True,
                )
                split[train_idx] = TRAIN_LABEL
                split[validation_idx] = VALIDATION_LABEL
                df[f"fold_{fold_id}"] = split
        else:
            kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=RANDOM_STATE)
            for fold_id, (train_validation_idx, test_idx) in enumerate(kf.split(np.arange(len(df)))):
                split = np.full(len(df), -1, dtype=np.int8)
                split[test_idx] = TEST_LABEL
                train_idx, validation_idx = train_test_split(
                    train_validation_idx,
                    test_size=1.0 / (self.train_to_validation_ratio + 1),
                    random_state=RANDOM_STATE,
                )
                split[train_idx] = TRAIN_LABEL
                split[validation_idx] = VALIDATION_LABEL
                df[f"fold_{fold_id}"] = split
        return df[df["type"] == 0], df[df["type"] == 1]


class TrainApplyFlow(NoOpStep):
    def __init__(
        self,
        name: str,
        kinematic: List[str],
        shower_shapes: List[str],
        weight_var: str,
        n_transforms: int,
        max_epoch: int,
        points_per_record: int,
        points_per_epoch: int,
        aux_nodes: int,
        aux_layers: int,
        n_splines_bins: int,
        initial_lr: float,
        batch_size: int,
        nfolds: int,
        weighted: bool,
        corr_suffix: str,
        fudged_prefix: str,
        outpath: str,
        variable_labels: Dict[str, str],
        binning: Dict[str, Binning],
        annotation: str,
        subannotation: str,
        ylabel: str,
        ratio_ylabel: str,
        legend_labels: Dict[str, str],
        kinematic_slices: Dict[str, List[float]],
        enabled: bool = True,
        standardize_step: Standardize | None = None,
        smooth_step: Smooth | None = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.kinematic = kinematic
        self.model_kinematic = kinematic + ["type"]
        self.shower_shapes = shower_shapes
        self.weight_var = weight_var
        self.n_transforms = n_transforms
        self.max_epoch = max_epoch
        self.points_per_record = points_per_record
        self.points_per_epoch = points_per_epoch
        self.aux_nodes = aux_nodes
        self.aux_layers = aux_layers
        self.n_splines_bins = n_splines_bins
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.weighted = weighted
        self.nfolds = nfolds
        self.corr_suffix = corr_suffix
        self.fudged_prefix = fudged_prefix
        self.outpath = outpath
        self.standardize_step = standardize_step
        self.smooth_step = smooth_step
        self.smoothing_config = self._build_smoothing_config()
        self.variable_labels = variable_labels
        self.binning = binning
        self.annotation = annotation
        self.subannotation = subannotation
        self.ylabel = ylabel
        self.ratio_ylabel = ratio_ylabel
        self.legend_labels = legend_labels
        self.kinematic_slices = kinematic_slices

    @property
    def used_variables(self) -> Iterable[str]:
        return set(self.kinematic) | set(self.shower_shapes) | {self.weight_var} | set(self.kinematic_slices)

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_index_shift = int(proto.index.max()) + 1 if len(proto.index) > 0 else 0
        target = target.copy()
        target.index += target_index_shift

        for fold_id in range(self.nfolds):
            outpath = setup_output_dir(os.path.join(self.outpath, f"fold_{fold_id}"), allow_recreate=True)
            df = pd.concat([proto, target], copy=False)
            df = df.sample(frac=1, random_state=RANDOM_STATE)

            train_df = df[df[f"fold_{fold_id}"] == TRAIN_LABEL]
            validation_df = df[df[f"fold_{fold_id}"] == VALIDATION_LABEL]

            trainer = Trainer(
                train_df=train_df,
                validation_df=validation_df,
                kinematic=self.model_kinematic,
                shower_shapes=self.shower_shapes,
                weight_var=self.weight_var,
                n_transforms=self.n_transforms,
                max_epoch=self.max_epoch,
                points_per_record=self.points_per_record,
                points_per_epoch=self.points_per_epoch,
                aux_nodes=self.aux_nodes,
                aux_layers=self.aux_layers,
                n_splines_bins=self.n_splines_bins,
                initial_lr=self.initial_lr,
                batch_size=self.batch_size,
                weighted=self.weighted,
                outpath=outpath,
            )
            flow = trainer.train()

            test_df = df[df[f"fold_{fold_id}"] == TEST_LABEL]
            applier = Applier(
                flow=flow,
                df=test_df,
                kinematic=self.model_kinematic,
                shower_shapes=self.shower_shapes,
                weight_var=self.weight_var,
                corr_suffix=self.corr_suffix,
                outpath=outpath,
            )
            applier.apply()
            proto_fold_corr = applier.correct()
            proto.loc[proto_fold_corr.index, proto_fold_corr.columns] = proto_fold_corr

            source_plot_df = proto.loc[proto_fold_corr.index, :].copy()
            target_plot_df = test_df[test_df["type"] == 1].copy()

            if self.standardize_step is not None:
                source_plot_df, target_plot_df = self.standardize_step.post(source_plot_df, target_plot_df)
            if self.smooth_step is not None:
                source_plot_df, target_plot_df = self.smooth_step.post(source_plot_df, target_plot_df)

            plot_corrected_hists(
                proto=source_plot_df,
                target=target_plot_df,
                variables=self.shower_shapes,
                weight_var=self.weight_var,
                variable_labels=self.variable_labels,
                binning=self.binning,
                annotation=self.annotation,
                subannotation=self.subannotation,
                ylabel=self.ylabel,
                ratio_ylabel=self.ratio_ylabel,
                legend_labels=self.legend_labels,
                corr_suffix=self.corr_suffix,
                outpath=setup_output_dir(os.path.join(outpath, "corrected"), allow_recreate=True),
                hint=f"fold_{fold_id}",
            )

            if self.fudged_prefix and all(var.replace(self.fudged_prefix, "") in source_plot_df.columns for var in self.shower_shapes):
                plot_corrected_fudged_hists(
                    proto=source_plot_df,
                    target=target_plot_df,
                    variables=self.shower_shapes,
                    weight_var=self.weight_var,
                    variable_labels=self.variable_labels,
                    binning=self.binning,
                    annotation=self.annotation,
                    subannotation=self.subannotation,
                    ylabel=self.ylabel,
                    ratio_ylabel=self.ratio_ylabel,
                    legend_labels=self.legend_labels,
                    corr_suffix=self.corr_suffix,
                    fudged_prefix=self.fudged_prefix,
                    outpath=setup_output_dir(os.path.join(outpath, "corrected_fudged"), allow_recreate=True),
                    hint=f"fold_{fold_id}",
                )

            plot_correlation_matrices(
                proto=source_plot_df,
                target=target_plot_df,
                variables=self.shower_shapes,
                weight_var=self.weight_var,
                corr_suffix=self.corr_suffix,
                variable_labels=self.variable_labels,
                outpath=setup_output_dir(os.path.join(outpath, "correlations"), allow_recreate=True),
            )

            if self.kinematic_slices:
                plot_corrected_hists_slices(
                    proto=source_plot_df,
                    target=target_plot_df,
                    variables=self.shower_shapes,
                    weight_var=self.weight_var,
                    corr_suffix=self.corr_suffix,
                    slices=self.kinematic_slices,
                    variable_labels=self.variable_labels,
                    binning=self.binning,
                    annotation=self.annotation,
                    subannotation=self.subannotation,
                    ylabel=self.ylabel,
                    ratio_ylabel=self.ratio_ylabel,
                    legend_labels=self.legend_labels,
                    outpath=setup_output_dir(os.path.join(outpath, "slices"), allow_recreate=True),
                    hint=f"fold_{fold_id}",
                )

        target.index -= target_index_shift
        return proto, target

    def _build_smoothing_config(self) -> SmoothingConfig | None:
        if self.smooth_step is None:
            return None
        eratio_idx = tuple(i for i, var in enumerate(self.shower_shapes) if var in self.smooth_step.eratio_variables)
        deltae_idx = tuple(i for i, var in enumerate(self.shower_shapes) if var in self.smooth_step.deltae_variables)
        if not eratio_idx and not deltae_idx:
            return None
        return SmoothingConfig(eratio_idx=eratio_idx, deltae_idx=deltae_idx)

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("not supported in post context")


class Shuffle(NoOpStep):
    def __init__(self, name: str, enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.enabled:
            return proto, target
        print("\033[1;36m[INFO]\033[95m Datasets are shuffled! \033[0m")
        return proto.sample(frac=1, random_state=RANDOM_STATE), target.sample(frac=1, random_state=RANDOM_STATE)

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise RuntimeError("not supported in post context")


class PlotCorrectedVariables(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: Iterable[str],
        weight_var: str,
        variable_labels: Dict[str, str],
        binning: Dict[str, Binning],
        annotation: str,
        subannotation: str,
        ylabel: str,
        ratio_ylabel: str,
        legend_labels: Dict[str, str],
        corr_suffix: str,
        outpath: str,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variables = list(variables)
        self.weight_var = weight_var
        self.binning = binning
        self.variable_labels = variable_labels
        self.annotation = annotation
        self.subannotation = subannotation
        self.ylabel = ylabel
        self.ratio_ylabel = ratio_ylabel
        self.legend_labels = legend_labels
        self.corr_suffix = corr_suffix
        self.outpath = setup_output_dir(os.path.join(outpath, name), allow_recreate=False)

        for var in self.variables:
            if var not in self.binning:
                raise ValueError(f"variable {var} not found in binning dict")
            if var not in self.variable_labels:
                raise ValueError(f"variable {var} not found in variable_labels dict")

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + [self.weight_var]

    def plot(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        plot_corrected_hists(
            proto=proto,
            target=target,
            variables=self.variables,
            weight_var=self.weight_var,
            variable_labels=self.variable_labels,
            binning=self.binning,
            annotation=self.annotation,
            subannotation=self.subannotation,
            ylabel=self.ylabel,
            ratio_ylabel=self.ratio_ylabel,
            legend_labels=self.legend_labels,
            corr_suffix=self.corr_suffix,
            outpath=setup_output_dir(os.path.join(self.outpath, "corrected"), allow_recreate=True),
        )

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target


class PlotCorrectedFudgedVariables(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: Iterable[str],
        weight_var: str,
        variable_labels: Dict[str, str],
        binning: Dict[str, Binning],
        annotation: str,
        subannotation: str,
        ylabel: str,
        ratio_ylabel: str,
        legend_labels: Dict[str, str],
        corr_suffix: str,
        fudged_prefix: str,
        outpath: str,
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variables = list(variables)
        self.weight_var = weight_var
        self.binning = binning
        self.variable_labels = variable_labels
        self.annotation = annotation
        self.subannotation = subannotation
        self.ylabel = ylabel
        self.ratio_ylabel = ratio_ylabel
        self.legend_labels = legend_labels
        self.corr_suffix = corr_suffix
        self.fudged_prefix = fudged_prefix
        self.outpath = setup_output_dir(os.path.join(outpath, name), allow_recreate=False)

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + [var.replace(self.fudged_prefix, "") for var in self.variables] + [self.weight_var]

    def plot(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        plot_corrected_fudged_hists(
            proto=proto,
            target=target,
            variables=self.variables,
            weight_var=self.weight_var,
            variable_labels=self.variable_labels,
            binning=self.binning,
            annotation=self.annotation,
            subannotation=self.subannotation,
            ylabel=self.ylabel,
            ratio_ylabel=self.ratio_ylabel,
            legend_labels=self.legend_labels,
            corr_suffix=self.corr_suffix,
            fudged_prefix=self.fudged_prefix,
            outpath=self.outpath,
        )

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.plot(proto, target)
        return proto, target


class ExportSource(NoOpStep):
    def __init__(self, name: str, outpath: str, filename: str, fmt: str = "hdf5", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.outpath = outpath
        self.filename = filename
        self.fmt = fmt

    def export(self, df: pd.DataFrame) -> None:
        base = f"{self.name}_{self.filename}" if self.filename else self.name
        suffix = {"hdf5": ".h5", "parquet": ".parquet", "root": ".root"}[self.fmt]
        output = Path(self.outpath) / (base if base.endswith(suffix) else f"{base}{suffix}")
        export_numeric_dataframe(df, output, fmt=self.fmt)

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.export(proto)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.export(proto)
        return proto, target


class PlotCorrelationMatrices(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: List[str],
        weight_var: str,
        corr_suffix: str,
        outpath: str,
        variable_labels: Dict[str, str],
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variable_labels = variable_labels
        self.variables = variables
        self.weight_var = weight_var
        self.corr_suffix = corr_suffix
        self.outpath = setup_output_dir(os.path.join(outpath, name), allow_recreate=False)

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + [self.weight_var]

    def plot(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        plot_correlation_matrices(
            proto=proto,
            target=target,
            variables=self.variables,
            weight_var=self.weight_var,
            corr_suffix=self.corr_suffix,
            variable_labels=self.variable_labels,
            outpath=self.outpath,
        )

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.plot(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.plot(proto, target)
        return proto, target


class PlotCorrectedVariablesSlices(NoOpStep):
    def __init__(
        self,
        name: str,
        variables: List[str],
        weight_var: str,
        corr_suffix: str,
        outpath: str,
        variable_labels: Dict[str, str],
        binning: Dict[str, Binning],
        annotation: str,
        subannotation: str,
        ylabel: str,
        ratio_ylabel: str,
        legend_labels: Dict[str, str],
        kinematic_slices: Dict[str, List[float]],
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.variable_labels = variable_labels
        self.variables = variables
        self.weight_var = weight_var
        self.corr_suffix = corr_suffix
        self.binning = binning
        self.annotation = annotation
        self.subannotation = subannotation
        self.ylabel = ylabel
        self.ratio_ylabel = ratio_ylabel
        self.legend_labels = legend_labels
        self.outpath = setup_output_dir(os.path.join(outpath, name), allow_recreate=False)
        self.kinematic_slices = kinematic_slices

    @property
    def used_variables(self) -> Iterable[str]:
        return self.variables + list(self.kinematic_slices) + [self.weight_var]

    def plot(self, proto: pd.DataFrame, target: pd.DataFrame) -> None:
        plot_corrected_hists_slices(
            proto=proto,
            target=target,
            variables=self.variables,
            weight_var=self.weight_var,
            corr_suffix=self.corr_suffix,
            slices=self.kinematic_slices,
            variable_labels=self.variable_labels,
            binning=self.binning,
            annotation=self.annotation,
            subannotation=self.subannotation,
            ylabel=self.ylabel,
            ratio_ylabel=self.ratio_ylabel,
            legend_labels=self.legend_labels,
            outpath=self.outpath,
        )

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.plot(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.enabled:
            self.plot(proto, target)
        return proto, target
