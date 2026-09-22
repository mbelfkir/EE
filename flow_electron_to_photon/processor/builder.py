from __future__ import annotations

from typing import Dict, Iterable, Tuple

from flow_electron_to_photon.processor.config import (
    ApplyFoldMarkupStep,
    ApplySavedFlowFoldsStep,
    CastToFloatStep,
    ExportSourceStep,
    NoWeightsStep,
    PlotCorrectedFudgedVariablesStep,
    PlotCorrectedVariablesSlicesStep,
    PlotCorrectedVariablesStep,
    PlotCorrelationMatricesStep,
    PlotVariablesStep,
    PrinterStep,
    QueryCutsStep,
    ShuffleStep,
    SmoothStep,
    SplitTrainTestValidationStep,
    StandardizeStep,
    Step,
    TrainApplyFlowStep,
    TruncateEventsStep,
)
from flow_electron_to_photon.processor.protocol import Step as ProtocolStep
from flow_electron_to_photon.processor.steps import (
    ApplyFoldMarkup,
    ApplySavedFlowFolds,
    CastToFloat,
    ExportSource,
    NoWeights,
    PlotCorrectedFudgedVariables,
    PlotCorrectedVariables,
    PlotCorrectedVariablesSlices,
    PlotCorrelationMatrices,
    PlotVariables,
    Printer,
    QueryCuts,
    Shuffle,
    Smooth,
    SplitTrainTestValidation,
    Standardize,
    TrainApplyFlow,
    TruncateEvents,
)


def build_steps(
    pre_steps: Iterable[Step],
    post_steps: Iterable[Step],
    outpath: str,
) -> Tuple[Iterable[ProtocolStep], Iterable[ProtocolStep]]:
    pre_steps_built: dict[str, ProtocolStep] = {}
    for step in pre_steps:
        if step.name in pre_steps_built:
            raise ValueError(f"Duplicate pre step found: {step.name}")
        pre_steps_built[step.name] = build_step(step, outpath, pre_steps_built)

    post_steps_built: dict[str, ProtocolStep] = {}
    for step in post_steps:
        if step.name in post_steps_built:
            raise ValueError(f"Duplicate post step found: {step.name}")

        post_candidate = build_step(step, outpath, pre_steps_built | post_steps_built)
        if step.name in pre_steps_built:
            pre = pre_steps_built[step.name]
            if not isinstance(post_candidate, pre.__class__):
                raise ValueError(f"Paired post step name mismatch: {step.name}")
            post_steps_built[step.name] = pre
            print(f"Matched pre and post steps with name: {step.name}")
            continue

        post_steps_built[step.name] = post_candidate

    return pre_steps_built.values(), post_steps_built.values()


def build_step(step: Step, outpath: str, steps: Dict[str, ProtocolStep]) -> ProtocolStep:
    if isinstance(step, QueryCutsStep):
        return QueryCuts(name=step.name, expressions=step.expressions, enabled=step.enabled)
    if isinstance(step, NoWeightsStep):
        return NoWeights(name=step.name, enabled=step.enabled, weight_var=step.weight_var)
    if isinstance(step, StandardizeStep):
        return Standardize(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            outpath=outpath,
            weighted=step.weighted,
            post_mapping=step.post_mapping,
            mode=step.mode,
            stats_path=step.stats_path,
        )
    if isinstance(step, TruncateEventsStep):
        return TruncateEvents(name=step.name, enabled=step.enabled, offset=step.offset, limit=step.limit)
    if isinstance(step, CastToFloatStep):
        return CastToFloat(name=step.name, enabled=step.enabled, variables=step.variables)
    if isinstance(step, SmoothStep):
        return Smooth(
            name=step.name,
            enabled=step.enabled,
            eratio_variables=step.eratio_variables,
            deltae_variables=step.deltae_variables,
        )
    if isinstance(step, PrinterStep):
        return Printer(name=step.name, enabled=step.enabled, text=step.text)
    if isinstance(step, PlotVariablesStep):
        return PlotVariables(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            variable_labels=step.variable_labels,
            binning=step.binning,
            annotation=step.annotation,
            subannotation=step.subannotation,
            ylabel=step.ylabel,
            ratio_ylabel=step.ratio_ylabel,
            legend_labels=step.legend_labels,
            outpath=outpath,
        )
    if isinstance(step, SplitTrainTestValidationStep):
        return SplitTrainTestValidation(
            name=step.name,
            enabled=step.enabled,
            train_to_validation_ratio=step.train_to_validation_ratio,
            nfolds=step.nfolds,
            fold_seed_variable=step.fold_seed_variable,
            split_mode=step.split_mode,
            test_fraction=step.test_fraction,
            stratify_by_type=step.stratify_by_type,
        )
    if isinstance(step, TrainApplyFlowStep):
        if not isinstance(standardize_step := steps.get(step.standardize_step_name), (Standardize, type(None))):
            raise ValueError("invalid standardize step type passed to train_apply_flow")
        if not isinstance(smooth_step := steps.get(step.smooth_step_name), (Smooth, type(None))):
            raise ValueError("invalid smooth step type passed to train_apply_flow")
        return TrainApplyFlow(
            name=step.name,
            enabled=step.enabled,
            kinematic=step.kinematic,
            shower_shapes=step.shower_shapes,
            weight_var=step.weight_var,
            n_transforms=step.n_transforms,
            max_epoch=step.max_epoch,
            points_per_record=step.points_per_record,
            points_per_epoch=step.points_per_epoch,
            aux_nodes=step.aux_nodes,
            aux_layers=step.aux_layers,
            n_splines_bins=step.n_splines_bins,
            initial_lr=step.initial_lr,
            batch_size=step.batch_size,
            weighted=step.weighted,
            nfolds=step.nfolds,
            corr_suffix=step.corr_suffix,
            fudged_prefix=step.fudged_prefix,
            variable_labels=step.variable_labels,
            binning=step.binning,
            annotation=step.annotation,
            subannotation=step.subannotation,
            ylabel=step.ylabel,
            ratio_ylabel=step.ratio_ylabel,
            legend_labels=step.legend_labels,
            kinematic_slices=step.kinematic_slices,
            outpath=outpath,
            standardize_step=standardize_step,
            smooth_step=smooth_step,
        )
    if isinstance(step, ShuffleStep):
        return Shuffle(name=step.name, enabled=step.enabled)
    if isinstance(step, PlotCorrectedVariablesStep):
        return PlotCorrectedVariables(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            variable_labels=step.variable_labels,
            binning=step.binning,
            annotation=step.annotation,
            subannotation=step.subannotation,
            ylabel=step.ylabel,
            ratio_ylabel=step.ratio_ylabel,
            legend_labels=step.legend_labels,
            corr_suffix=step.corr_suffix,
            outpath=outpath,
        )
    if isinstance(step, PlotCorrectedFudgedVariablesStep):
        return PlotCorrectedFudgedVariables(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            variable_labels=step.variable_labels,
            binning=step.binning,
            annotation=step.annotation,
            subannotation=step.subannotation,
            ylabel=step.ylabel,
            ratio_ylabel=step.ratio_ylabel,
            legend_labels=step.legend_labels,
            corr_suffix=step.corr_suffix,
            fudged_prefix=step.fudged_prefix,
            outpath=outpath,
        )
    if isinstance(step, ExportSourceStep):
        return ExportSource(
            name=step.name,
            enabled=step.enabled,
            outpath=outpath,
            filename=step.filename,
            fmt=step.format,
        )
    if isinstance(step, PlotCorrelationMatricesStep):
        return PlotCorrelationMatrices(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            outpath=outpath,
            variable_labels=step.variable_labels,
            corr_suffix=step.corr_suffix,
        )
    if isinstance(step, PlotCorrectedVariablesSlicesStep):
        return PlotCorrectedVariablesSlices(
            name=step.name,
            enabled=step.enabled,
            variables=step.variables,
            weight_var=step.weight_var,
            corr_suffix=step.corr_suffix,
            outpath=outpath,
            variable_labels=step.variable_labels,
            binning=step.binning,
            annotation=step.annotation,
            subannotation=step.subannotation,
            ylabel=step.ylabel,
            ratio_ylabel=step.ratio_ylabel,
            legend_labels=step.legend_labels,
            kinematic_slices=step.kinematic_slices,
        )
    if isinstance(step, ApplyFoldMarkupStep):
        return ApplyFoldMarkup(
            name=step.name,
            enabled=step.enabled,
            nfolds=step.nfolds,
            fold_seed_variable=step.fold_seed_variable,
            column=step.column,
            apply_to_target=step.apply_to_target,
        )
    if isinstance(step, ApplySavedFlowFoldsStep):
        return ApplySavedFlowFolds(
            name=step.name,
            enabled=step.enabled,
            nfolds=step.nfolds,
            model_paths=step.model_paths,
            kinematic=step.kinematic,
            shower_shapes=step.shower_shapes,
            weight_var=step.weight_var,
            corr_suffix=step.corr_suffix,
            fold_column=step.fold_column,
            batch_size=step.batch_size,
            outpath=outpath,
        )
    raise ValueError(f"Unsupported step configuration type: {type(step)!r}")
