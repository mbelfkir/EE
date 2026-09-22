from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import hist
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zuko
from functools import partial

from flow_electron_to_photon.misc.system import setup_output_dir
from flow_electron_to_photon.plotting.hists import plot_bases
from flow_electron_to_photon.plotting.loss import plot_loss_batch, plot_loss_epoch


torch.set_num_threads(8)

TRAIN_LABEL = 0
VALIDATION_LABEL = 1
TEST_LABEL = 2


def build_nsf(
    features: int,
    context: int = 0,
    bins: int = 8,
    **kwargs: Any,
) -> zuko.flows.MAF:
    return zuko.flows.MAF(
        features=features,
        context=context,
        univariate=partial(zuko.transforms.MonotonicRQSTransform, bound=5.0, slope=1e-2),
        shapes=[(bins,), (bins,), (bins - 1,)],
        **kwargs,
    )


class SmoothingConfig(NamedTuple):
    eratio_idx: Tuple[int, ...]
    deltae_idx: Tuple[int, ...]

    @property
    def has_smoothing(self) -> bool:
        return bool(self.eratio_idx or self.deltae_idx)


class TensorSmoother:
    @staticmethod
    def _pseudo_triangular_leftmode_like(ref: torch.Tensor, left: float, shift: float, seed: float) -> torch.Tensor:
        n = ref.shape[0]
        i = torch.arange(n, device=ref.device, dtype=ref.dtype) + 1.0
        x = torch.sin(i * 12.3456 + seed) * 43758.5453
        u = x - torch.floor(x)
        return left + shift * torch.sqrt(u)

    @classmethod
    def smooth_deltae_(cls, tensor: torch.Tensor, index: int, eps: float = 1e-3) -> None:
        col = tensor[:, index]
        mask = col == 0
        samples = cls._pseudo_triangular_leftmode_like(col, left=1.0, shift=1.0, seed=0.12345 + float(index) * 0.01)
        col = torch.where(mask, samples, col + 3.0)
        tensor[:, index] = torch.log(col + eps)

    @classmethod
    def smooth_eratio_(cls, tensor: torch.Tensor, index: int, shift: float = 0.1, eps: float = 1e-3) -> None:
        col = tensor[:, index]
        left = 1.0 + shift
        mask = col >= 1.0
        samples = cls._pseudo_triangular_leftmode_like(col, left=left, shift=shift, seed=0.54321 + float(index) * 0.01)
        col = torch.where(mask, samples, col)
        tensor[:, index] = torch.log(col + eps)

    @classmethod
    def desmooth_eratio_(cls, tensor: torch.Tensor, index: int, eps: float = 1e-3) -> None:
        col = torch.exp(tensor[:, index]) - eps
        tensor[:, index] = torch.where(col > 1.0, col.new_tensor(1.0), col)

    @classmethod
    def desmooth_deltae_(cls, tensor: torch.Tensor, index: int, eps: float = 1e-3) -> None:
        col = torch.exp(tensor[:, index]) - eps
        mask_small = col < 3.0
        tensor[:, index] = torch.where(mask_small, col.new_zeros(()), col - 3.0)


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        shower_shapes: List[str],
        kinematic: List[str],
        weight_var: str,
    ):
        self.ss = torch.tensor(df[shower_shapes].to_numpy(dtype="float32"), dtype=torch.float32)
        self.kinematic = torch.tensor(df[kinematic].to_numpy(dtype="float32"), dtype=torch.float32)
        self.weights = torch.tensor(df[weight_var].to_numpy(dtype="float32"), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ss)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ss[idx], self.kinematic[idx], self.weights[idx]


class TrainStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss: float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Scheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        steps_per_cycle: int,
        eta_min: float = 0.0,
        epoch_decay: float = 0.8,
        factor: float = 0.5,
        patience: int = 5,
        min_delta: float = 0.0,
        min_scale: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.steps_per_cycle = max(int(steps_per_cycle), 1)
        self.epoch_decay = epoch_decay
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_scale = min_scale
        self.best = float("inf")
        self.bad = 0
        self.scale = 1.0
        self.steps = 0

    def step_epoch(self, metric: float) -> None:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.scale = max(self.scale * self.factor, self.min_scale)
                self.bad = 0

    def step_batch(self) -> float:
        lr = self.lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.steps += 1
        if self.steps % self.steps_per_cycle == 0:
            self.initial_lr *= self.epoch_decay
        return lr

    @property
    def lr(self) -> float:
        cycle_step = self.steps % self.steps_per_cycle
        if self.steps_per_cycle == 1:
            cos_factor = 1.0
        else:
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_step / (self.steps_per_cycle - 1)))
        return (self.eta_min + (self.initial_lr - self.eta_min) * cos_factor) * self.scale


def save_flow_checkpoint(
    flow: zuko.flows.MAF,
    path: str | Path,
    *,
    kinematic: List[str],
    shower_shapes: List[str],
    n_transforms: int,
    aux_nodes: int,
    aux_layers: int,
    n_splines_bins: int,
) -> None:
    payload = {
        "state_dict": flow.state_dict(),
        "kinematic": list(kinematic),
        "pure_kinematic": [var for var in kinematic if var != "type"],
        "shower_shapes": list(shower_shapes),
        "n_transforms": int(n_transforms),
        "aux_nodes": int(aux_nodes),
        "aux_layers": int(aux_layers),
        "n_splines_bins": int(n_splines_bins),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_flow_from_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[zuko.flows.MAF, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    flow = build_nsf(
        len(payload["shower_shapes"]),
        len(payload["kinematic"]),
        transforms=payload["n_transforms"],
        bins=payload["n_splines_bins"],
        hidden_features=[payload["aux_nodes"]] * payload["aux_layers"],
    )
    flow.load_state_dict(payload["state_dict"])
    flow.eval()
    return flow, payload


class Trainer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame | None,
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
        weighted: bool,
        outpath: str,
    ):
        self.train_df = train_df
        self.validation_df = validation_df
        self.kinematic = kinematic
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
        self.outpath = outpath

        self.training_loss_values: list[float] = []
        self.validation_loss_values: list[float] = []
        self.epoch_lr: list[float | None] = []
        self.batch_lr: list[float] = []
        self.training_loss_values_batch: list[float] = []
        self.validation_loss_values_batch: list[float] = []
        self.training_l2_values_batch: list[float] = []
        self.validation_l2_values_batch: list[float] = []

        self.batch_loss_prunning = max(int(len(self.train_df) / max(self.batch_size, 1) / max(self.points_per_epoch, 1)), 1)
        self.has_validation = self.validation_df is not None and len(self.validation_df) > 0
        self.train_sample_idx, self.validation_sample_idx = self.idxs

    @property
    def idxs(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.has_validation:
            fix_n = min(self.points_per_record, len(self.train_df), len(self.validation_df))
            return torch.randint(0, len(self.train_df), (fix_n,)), torch.randint(0, len(self.validation_df), (fix_n,))

        fix_n = min(self.points_per_record, len(self.train_df))
        return torch.randint(0, len(self.train_df), (fix_n,)), None

    def train(self) -> zuko.flows.MAF:
        print("\n\033[1;36m[INFO]\033[1;92m 🌐 Training has been started...\033[0m")

        self.training_dataset = CustomDataset(self.train_df, self.shower_shapes, self.kinematic, self.weight_var)
        self.validation_dataset = (
            CustomDataset(self.validation_df, self.shower_shapes, self.kinematic, self.weight_var)
            if self.has_validation and self.validation_df is not None
            else None
        )
        dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)

        self.flow = build_nsf(
            len(self.shower_shapes),
            len(self.kinematic),
            transforms=self.n_transforms,
            bins=self.n_splines_bins,
            hidden_features=[self.aux_nodes] * self.aux_layers,
        )
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.initial_lr)
        scheduler = Scheduler(
            optimizer,
            self.initial_lr,
            steps_per_cycle=len(dataloader),
            eta_min=self.initial_lr * 0.01,
        )
        stop_train = TrainStopper(patience=10, min_delta=0.0) if self.has_validation else None

        lr = None
        try:
            for epoch in range(self.max_epoch):
                self.flow.train()
                for batch_idx, (ss, kin, w) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
                    lr = scheduler.step_batch()
                    optimizer.zero_grad()
                    loss = self.compute_loss_grad(kin, ss, w)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                    optimizer.step()

                    if batch_idx % self.batch_loss_prunning == 0:
                        self.flow.eval()
                        self.record_loss_batch()
                        self.batch_lr.append(lr)
                        plot_loss_batch(
                            training_loss=self.training_loss_values_batch,
                            validation_loss=self.validation_loss_values_batch,
                            training_l2=self.training_l2_values_batch,
                            validation_l2=self.validation_l2_values_batch,
                            learning_rate=self.batch_lr,
                            batch_loss_prunning=self.batch_loss_prunning,
                            outpath=self.outpath,
                        )
                        self.flow.train()

                self.dump_model(epoch)

                with torch.no_grad():
                    self.flow.eval()
                    training_loss, validation_loss = self.record_loss_epoch()
                    scheduler.step_epoch(validation_loss if self.has_validation else training_loss)
                    self.epoch_lr.append(lr)
                    plot_loss_epoch(
                        training_loss=self.training_loss_values,
                        validation_loss=self.validation_loss_values,
                        learning_rate=self.epoch_lr,
                        outpath=self.outpath,
                    )
                    print(
                        f"\n\033[1;34m ⭐️  Epoch: {epoch}, \033[0m"
                        f"\033[1;32mTraining loss: {training_loss:.3f}, \033[0m"
                        + (
                            f"\033[1;31mValidation loss: {validation_loss:.3f} ⭐️\033[0m"
                            if self.has_validation
                            else "\033[1;31mValidation loss: n/a (holdout mode) ⭐️\033[0m"
                        )
                    )
                    if stop_train is not None and stop_train.early_stop(validation_loss):
                        best_id, best_loss = self.load_best_model()
                        print(f"\n\033[1;36m[INFO]\033[1;93m Lowest validation loss {best_loss:.6f} at epoch {best_id}\033[0m")
                        break
        except KeyboardInterrupt:
            print("\n\033[1;33m[WARNING]\033[0m Training interrupted by user.")

        best_id, best_loss = self.load_best_model()
        if self.has_validation:
            print(f"\n\033[1;36m[INFO]\033[1;93m Lowest validation loss {best_loss:.6f} at epoch {best_id}\033[0m")
        else:
            print(f"\n\033[1;36m[INFO]\033[1;93m Holdout mode saved final epoch {best_id} as best_model.pt\033[0m")
        return self.flow

    @torch.no_grad()
    def compute_l2(self, kin: torch.Tensor, ss: torch.Tensor, w: torch.Tensor) -> float:
        mask = kin[:, -1] == 0
        if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
            return float("nan")

        base_source = self.flow(kin[mask]).transform(ss[mask]).detach().numpy()
        base_target = self.flow(kin[~mask]).transform(ss[~mask]).detach().numpy()

        w_source = (w[mask] / w[mask].sum()).detach().numpy()
        w_target = (w[~mask] / w[~mask].sum()).detach().numpy()

        l2 = 0.0
        for index, var in enumerate(self.shower_shapes):
            base_source_hist = hist.new.Reg(200, -10, 10, name=var, label=var, overflow=False, underflow=False).Double()
            base_target_hist = hist.new.Reg(200, -10, 10, name=var, label=var, overflow=False, underflow=False).Double()
            base_source_hist.fill(base_source[:, index], weight=w_source)
            base_target_hist.fill(base_target[:, index], weight=w_target)
            l2 += float(np.sum((base_source_hist.values() - base_target_hist.values()) ** 2) ** 0.5)
        return l2

    def compute_loss_grad(self, kin: torch.Tensor, ss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        std = float(w.std().item()) if w.numel() > 1 else 0.0
        threshold = max(5.0 * std, 1.0)
        w = w.clamp(-threshold, threshold)
        loss = -self.flow(kin).log_prob(ss)
        return (loss * w).sum() / w.sum() if self.weighted else loss.mean()

    @torch.no_grad()
    def compute_loss(self, kin: torch.Tensor, ss: torch.Tensor, w: torch.Tensor) -> float:
        return float(self.compute_loss_grad(kin, ss, w))

    @torch.no_grad()
    def compute_random_batch_loss(self) -> Tuple[float, float]:
        training_loss = self.compute_loss(
            self.training_dataset.kinematic[self.train_sample_idx],
            self.training_dataset.ss[self.train_sample_idx],
            self.training_dataset.weights[self.train_sample_idx],
        )
        if not self.has_validation or self.validation_dataset is None or self.validation_sample_idx is None:
            return training_loss, float("nan")
        validation_loss = self.compute_loss(
            self.validation_dataset.kinematic[self.validation_sample_idx],
            self.validation_dataset.ss[self.validation_sample_idx],
            self.validation_dataset.weights[self.validation_sample_idx],
        )
        return training_loss, validation_loss

    @torch.no_grad()
    def compute_random_batch_l2(self) -> Tuple[float, float]:
        training_l2 = self.compute_l2(
            self.training_dataset.kinematic[self.train_sample_idx],
            self.training_dataset.ss[self.train_sample_idx],
            self.training_dataset.weights[self.train_sample_idx],
        )
        if not self.has_validation or self.validation_dataset is None or self.validation_sample_idx is None:
            return training_l2, float("nan")
        validation_l2 = self.compute_l2(
            self.validation_dataset.kinematic[self.validation_sample_idx],
            self.validation_dataset.ss[self.validation_sample_idx],
            self.validation_dataset.weights[self.validation_sample_idx],
        )
        return training_l2, validation_l2

    @torch.no_grad()
    def record_loss_batch(self) -> Tuple[float, float]:
        train_loss, validation_loss = self.compute_random_batch_loss()
        train_l2, validation_l2 = self.compute_random_batch_l2()
        self.training_loss_values_batch.append(train_loss)
        self.validation_loss_values_batch.append(validation_loss)
        self.training_l2_values_batch.append(train_l2)
        self.validation_l2_values_batch.append(validation_l2)
        return train_loss, validation_loss

    @torch.no_grad()
    def record_loss_epoch(self) -> Tuple[float, float]:
        train_loss, validation_loss = self.compute_random_batch_loss()
        self.training_loss_values.append(train_loss)
        self.validation_loss_values.append(validation_loss)
        return train_loss, validation_loss

    def load_best_model(self) -> tuple[int, float]:
        if self.has_validation and self.validation_loss_values:
            best_id = int(np.nanargmin(np.asarray(self.validation_loss_values, dtype=float)))
            best_loss = float(np.nanmin(np.asarray(self.validation_loss_values, dtype=float)))
        else:
            best_id = max(len(self.training_loss_values) - 1, 0)
            best_loss = float(self.training_loss_values[best_id]) if self.training_loss_values else float("nan")
        best_epoch_path = os.path.join(self.outpath, f"models/epoch_{best_id}.pt")
        flow, _ = load_flow_from_checkpoint(best_epoch_path)
        self.flow.load_state_dict(flow.state_dict())
        save_flow_checkpoint(
            self.flow,
            os.path.join(self.outpath, "models/best_model.pt"),
            kinematic=self.kinematic,
            shower_shapes=self.shower_shapes,
            n_transforms=self.n_transforms,
            aux_nodes=self.aux_nodes,
            aux_layers=self.aux_layers,
            n_splines_bins=self.n_splines_bins,
        )
        return best_id, best_loss

    def dump_model(self, epoch: int) -> None:
        save_flow_checkpoint(
            self.flow,
            os.path.join(self.outpath, f"models/epoch_{epoch}.pt"),
            kinematic=self.kinematic,
            shower_shapes=self.shower_shapes,
            n_transforms=self.n_transforms,
            aux_nodes=self.aux_nodes,
            aux_layers=self.aux_layers,
            n_splines_bins=self.n_splines_bins,
        )


class Applier:
    def __init__(
        self,
        flow: zuko.flows.MAF,
        df: pd.DataFrame,
        kinematic: List[str],
        shower_shapes: List[str],
        weight_var: str,
        corr_suffix: str,
        outpath: str,
    ):
        self.flow = flow.eval()
        self.df = df
        self.kinematic = kinematic
        self.shower_shapes = shower_shapes
        self.weight_var = weight_var
        self.outpath = outpath
        self.dataset = CustomDataset(self.df, self.shower_shapes, self.kinematic, self.weight_var)
        self.kin = self.dataset.kinematic
        self.ss = self.dataset.ss
        self.w = self.dataset.weights
        self.corr_suffix = corr_suffix

    @torch.no_grad()
    def apply_batched(self, kin: torch.Tensor, ss: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        chunks = []
        for index in range(0, kin.shape[0], batch_size):
            chunks.append(self.flow(kin[index:index + batch_size]).transform(ss[index:index + batch_size]))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def apply(self) -> None:
        mask = self.kin[:, -1] == 0
        if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
            return

        base_source = self.apply_batched(self.kin[mask], self.ss[mask]).detach().numpy()
        base_target = self.apply_batched(self.kin[~mask], self.ss[~mask]).detach().numpy()

        weight_source = (self.w[mask] / self.w[mask].sum()).detach().numpy()
        weight_target = (self.w[~mask] / self.w[~mask].sum()).detach().numpy()

        outpath = setup_output_dir(os.path.join(self.outpath, "bases"), allow_recreate=True)
        for index, var in enumerate(self.shower_shapes):
            base_source_hist = hist.new.Reg(200, -10, 10, name=var, label=var, overflow=False, underflow=False).Double()
            base_target_hist = hist.new.Reg(200, -10, 10, name=var, label=var, overflow=False, underflow=False).Double()
            base_source_hist.fill(base_source[:, index], weight=weight_source)
            base_target_hist.fill(base_target[:, index], weight=weight_target)
            centers = base_source_hist.axes[0].centers
            hist_source = base_source_hist.values()
            hist_target = base_target_hist.values()
            if hist_source.sum() > 0:
                hist_source = hist_source / hist_source.sum()
            if hist_target.sum() > 0:
                hist_target = hist_target / hist_target.sum()
            score = float(np.sum((hist_source - hist_target) ** 2))
            print(f"\033[1;36m[INFO]\033[93m Variable {var}: base-space histogram L2 diff = {score ** 0.5:.5e}\033[0m")
            plot_bases(hist_source, hist_target, centers, score, var, outpath)

    @torch.no_grad()
    def correct_batched(self, kin: torch.Tensor, ss: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        chunks = []
        for index in tqdm(range(0, kin.shape[0], batch_size), desc="Correcting batches", leave=False):
            kin_batch = kin[index:index + batch_size]
            ss_batch = ss[index:index + batch_size]
            mask_ok = (ss_batch.abs() < 5).all(dim=1)
            base = self.flow(kin_batch).transform(ss_batch)
            mask_ok &= (base.abs() < 5).all(dim=1)
            kin_fake = torch.cat([kin_batch[:, :-1], torch.ones(kin_batch.size(0), 1)], dim=1)
            if int(mask_ok.sum()) > 0:
                ss_batch = ss_batch.clone()
                ss_batch[mask_ok] = self.flow(kin_fake[mask_ok]).transform.inv(base[mask_ok])
            chunks.append(ss_batch)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def correct(self) -> pd.DataFrame:
        mask = self.kin[:, -1] == 0
        ss_corr = self.correct_batched(self.kin[mask], self.ss[mask])
        return pd.DataFrame(
            ss_corr.numpy(),
            index=self.df.index[mask.numpy()],
            columns=[f"{variable}{self.corr_suffix}" for variable in self.shower_shapes],
        )
