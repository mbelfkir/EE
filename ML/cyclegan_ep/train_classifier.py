from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/train_classifier.py requires torch. Install the packages listed in ML/requirements.txt."
    ) from exc

from dataset import (
    DEFAULT_IMAGE_TRANSFORM,
    DEFAULT_MIN_WINDOW_ENERGY_GEV,
    DEFAULT_USE_STANDARDIZATION,
    ElectronPhotonClassificationDataset,
    compute_normalization_scale,
    prepare_datasets,
    save_dataset_manifest,
)
from models import GlobalDiscriminator
from utils import append_csv_row, build_linear_scheduler, choose_device, ensure_dir, init_weights, save_json, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = ML_DIR / "runs" / "electron_photon_classifier_11x11"
DEFAULT_ETA_WINDOW_SIZE = 11
DEFAULT_PHI_WINDOW_SIZE = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN classifier to distinguish electron from photon calorimeter "
            "windows using the same MSE target loss as the CycleGAN image discriminator."
        )
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing the ROOT calorimeter files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and logs")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--use-standardization", dest="use_standardization", action="store_true")
    parser.add_argument("--no-standardization", dest="use_standardization", action="store_false")
    parser.set_defaults(use_standardization=DEFAULT_USE_STANDARDIZATION)
    parser.add_argument("--min-window-energy-gev", type=float, default=DEFAULT_MIN_WINDOW_ENERGY_GEV)
    parser.add_argument("--image-transform", choices=["none", "cbrt"], default=DEFAULT_IMAGE_TRANSFORM)
    parser.add_argument("--eta-window-size", type=int, default=DEFAULT_ETA_WINDOW_SIZE)
    parser.add_argument("--phi-window-size", type=int, default=DEFAULT_PHI_WINDOW_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs at constant learning rate")
    parser.add_argument("--n-epochs-decay", type=int, default=0, help="Number of linearly decaying epochs")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--ndf", type=int, default=64, help="Base channel count for the discriminator-style CNN")
    parser.add_argument("--disc-layers", type=int, default=2, help="Number of stride-2 convolution blocks")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Threshold applied to the scalar score")
    parser.add_argument(
        "--max-train-samples-per-domain",
        type=int,
        default=0,
        help="Optional cap per domain for the training split; <=0 uses all samples",
    )
    parser.add_argument(
        "--max-val-samples-per-domain",
        type=int,
        default=0,
        help="Optional cap per domain for the validation split; <=0 uses all samples",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/last_model.pt if it exists")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_subset(samples: Sequence, max_count: int, seed: int) -> list:
    items = list(samples)
    if max_count <= 0 or len(items) <= max_count:
        return items
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(items), generator=generator).tolist()
    chosen = sorted(permutation[:max_count])
    return [items[index] for index in chosen]


def classifier_loss(prediction: Tensor, target: Tensor, criterion: nn.Module) -> Tensor:
    return criterion(prediction, target.expand_as(prediction))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    score_threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_examples = 0
    total_loss = 0.0
    total_correct = 0.0
    electron_total = 0
    photon_total = 0
    electron_correct = 0.0
    photon_correct = 0.0
    electron_score_sum = 0.0
    photon_score_sum = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            prediction = model(images)
            loss = classifier_loss(prediction, labels, criterion)
            if is_training:
                loss.backward()
                optimizer.step()

        scores = prediction.detach().view(-1)
        label_values = labels.detach().view(-1)
        predicted_labels = (scores >= score_threshold).to(label_values.dtype)
        correct = predicted_labels.eq(label_values)

        batch_size = int(label_values.numel())
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += float(correct.sum().item())

        electron_mask = label_values == 0.0
        photon_mask = label_values == 1.0
        if electron_mask.any():
            electron_count = int(electron_mask.sum().item())
            electron_total += electron_count
            electron_correct += float(correct[electron_mask].sum().item())
            electron_score_sum += float(scores[electron_mask].sum().item())
        if photon_mask.any():
            photon_count = int(photon_mask.sum().item())
            photon_total += photon_count
            photon_correct += float(correct[photon_mask].sum().item())
            photon_score_sum += float(scores[photon_mask].sum().item())

    if total_examples == 0:
        raise ValueError("Received an empty dataloader.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "electron_accuracy": electron_correct / max(electron_total, 1),
        "photon_accuracy": photon_correct / max(photon_total, 1),
        "electron_score_mean": electron_score_sum / max(electron_total, 1),
        "photon_score_mean": photon_score_sum / max(photon_total, 1),
        "examples": float(total_examples),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    last_model_path = output_dir / "last_model.pt"
    best_model_path = output_dir / "best_model.pt"
    log_path = output_dir / "training_log.csv"

    print(
        f"Preparing classification dataset from {Path(args.data_dir).resolve()} with "
        f"windows {args.phi_window_size}x{args.eta_window_size}",
        flush=True,
    )
    bundle = prepare_datasets(
        data_dir=Path(args.data_dir).resolve(),
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
        use_standardization=bool(args.use_standardization),
        backend="cnn",
        eta_window_size=int(args.eta_window_size),
        phi_window_size=int(args.phi_window_size),
        min_window_energy_gev=float(args.min_window_energy_gev),
        image_transform=str(args.image_transform),
        representation="image",
    )
    save_dataset_manifest(bundle, output_dir)
    symmetric_range = bundle.standardization is not None
    normalization_scale = compute_normalization_scale(
        [bundle.electron_train, bundle.photon_train],
        symmetric_range=symmetric_range,
    )

    train_electron = select_subset(bundle.electron_train, int(args.max_train_samples_per_domain), int(args.seed))
    train_photon = select_subset(bundle.photon_train, int(args.max_train_samples_per_domain), int(args.seed) + 1)
    if bundle.electron_val and bundle.photon_val:
        val_split_name = "val"
        val_electron_source = bundle.electron_val
        val_photon_source = bundle.photon_val
    else:
        val_split_name = "train"
        val_electron_source = bundle.electron_train
        val_photon_source = bundle.photon_train
    val_electron = select_subset(val_electron_source, int(args.max_val_samples_per_domain), int(args.seed) + 2)
    val_photon = select_subset(val_photon_source, int(args.max_val_samples_per_domain), int(args.seed) + 3)

    print(
        "Loaded samples: "
        f"train_electron={len(train_electron)}, "
        f"train_photon={len(train_photon)}, "
        f"{val_split_name}_electron={len(val_electron)}, "
        f"{val_split_name}_photon={len(val_photon)}",
        flush=True,
    )

    train_dataset = ElectronPhotonClassificationDataset(
        electron_samples=train_electron,
        photon_samples=train_photon,
        normalization_scale=normalization_scale,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        symmetric_range=symmetric_range,
    )
    val_dataset = ElectronPhotonClassificationDataset(
        electron_samples=val_electron,
        photon_samples=val_photon,
        normalization_scale=normalization_scale,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        symmetric_range=symmetric_range,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = GlobalDiscriminator(
        input_nc=1,
        ndf=int(args.ndf),
        n_layers=int(args.disc_layers),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = build_linear_scheduler(optimizer, int(args.n_epochs), int(args.n_epochs_decay))

    start_epoch = 1
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    if args.resume:
        if not last_model_path.exists():
            raise FileNotFoundError(f"--resume was requested but {last_model_path} does not exist.")
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_accuracy = float(checkpoint.get("best_val_accuracy", best_val_accuracy))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        print(f"Resuming from epoch {start_epoch}", flush=True)
    else:
        model.apply(init_weights)

    config = vars(args).copy()
    config["data_dir"] = str(Path(args.data_dir).resolve())
    config["output_dir"] = str(output_dir)
    config["normalization_scale"] = float(normalization_scale)
    config["symmetric_normalization"] = bool(symmetric_range)
    config["label_convention"] = {"electron": 0.0, "photon": 1.0}
    config["model_description"] = (
        "Global discriminator-style CNN classifier built from the CycleGAN image "
        f"PatchGAN trunk with AdaptiveAvgPool2d, ndf={args.ndf}, disc_layers={args.disc_layers}, "
        "loss=nn.MSELoss against 0/1 class targets."
    )
    save_json(config, output_dir / "config.json")
    save_json(
        {
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "val_split_used": val_split_name,
            "num_parameters": count_parameters(model),
        },
        output_dir / "model_summary.json",
    )

    total_epochs = int(args.n_epochs) + int(args.n_epochs_decay)
    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            score_threshold=float(args.score_threshold),
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            score_threshold=float(args.score_threshold),
            optimizer=None,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        log_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_electron_accuracy": train_metrics["electron_accuracy"],
            "train_photon_accuracy": train_metrics["photon_accuracy"],
            "train_electron_score_mean": train_metrics["electron_score_mean"],
            "train_photon_score_mean": train_metrics["photon_score_mean"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_electron_accuracy": val_metrics["electron_accuracy"],
            "val_photon_accuracy": val_metrics["photon_accuracy"],
            "val_electron_score_mean": val_metrics["electron_score_mean"],
            "val_photon_score_mean": val_metrics["photon_score_mean"],
        }
        append_csv_row(log_path, log_row)

        print(
            f"[Epoch {epoch:03d}/{total_epochs:03d}] "
            f"[train_loss: {train_metrics['loss']:.4f}] "
            f"[train_acc: {train_metrics['accuracy']:.4f}] "
            f"[val_loss: {val_metrics['loss']:.4f}] "
            f"[val_acc: {val_metrics['accuracy']:.4f}] "
            f"[val_e: {val_metrics['electron_accuracy']:.4f}] "
            f"[val_g: {val_metrics['photon_accuracy']:.4f}]",
            flush=True,
        )

        is_best = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                abs(val_metrics["accuracy"] - best_val_accuracy) < 1e-12
                and val_metrics["loss"] < best_val_loss
            )
        )
        if is_best:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "latest_train_metrics": train_metrics,
            "latest_val_metrics": val_metrics,
        }
        torch.save(checkpoint_state, last_model_path)
        if is_best:
            torch.save(checkpoint_state, best_model_path)

        save_json(
            {
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "latest_epoch": epoch,
                "latest_train_metrics": train_metrics,
                "latest_val_metrics": val_metrics,
            },
            output_dir / "metrics.json",
        )

        scheduler.step()


if __name__ == "__main__":
    main()
