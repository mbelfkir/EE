from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flow.config import build_flow_parser, canonicalise_flow_args
from flow.datasets import IMAGE_SCALING_MODE, SingleDomainCalorimeterDataset, prepare_datasets, save_dataset_manifest, select_fixed_samples
from flow.losses import latent_moment_loss, maximum_mean_discrepancy
from flow.models import ConditionalLatentFlow, build_autoencoder_from_config
from flow.plotting import plot_training_history
from flow.utils import (
    build_linear_scheduler,
    choose_device,
    ensure_dir,
    infinite_loader,
    latest_checkpoint_path,
    make_torch_generator,
    save_json,
    seed_worker,
    serialise_config,
    set_requires_grad,
    set_seed,
    specified_cli_flags,
)
from flow.validate import run_flow_validation


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
    )


def apply_autoencoder_config_defaults(args, autoencoder_config: dict[str, object], cli_flags: set[str]) -> None:
    fallback_map = {
        "--data-dir": "data_dir",
        "--train-fraction": "train_fraction",
        "--min-window-energy-gev": "min_window_energy_gev",
        "--image-transform": "image_transform",
        "--eta-window-size": "eta_window_size",
        "--phi-window-size": "phi_window_size",
        "--latent-dim": "latent_dim",
        "--encoder-channels": "encoder_channels",
        "--activation": "activation",
        "--norm": "norm",
        "--output-activation": "output_activation",
    }
    for flag, key in fallback_map.items():
        if flag not in cli_flags and key in autoencoder_config:
            setattr(args, key, autoencoder_config[key])

    if "--use-standardization" not in cli_flags and "--no-standardization" not in cli_flags and "use_standardization" in autoencoder_config:
        args.use_standardization = bool(autoencoder_config["use_standardization"])
    elif "--use-standardization" not in cli_flags and "--no-standardization" not in cli_flags and autoencoder_config.get("image_scaling") == IMAGE_SCALING_MODE:
        args.use_standardization = True


def main() -> None:
    args = canonicalise_flow_args(build_flow_parser().parse_args())
    cli_flags = specified_cli_flags()
    args.autoencoder_checkpoint = str(Path(args.autoencoder_checkpoint).resolve())

    autoencoder_checkpoint = torch.load(args.autoencoder_checkpoint, map_location="cpu")
    autoencoder_config = dict(autoencoder_checkpoint.get("config", {}))
    apply_autoencoder_config_defaults(args, autoencoder_config, cli_flags)
    args = canonicalise_flow_args(args)
    checkpoint_latent_dim = int(autoencoder_config.get("latent_dim", args.latent_dim))
    if int(args.latent_dim) != checkpoint_latent_dim:
        raise ValueError(
            "Stage-2 latent_dim must match the trained autoencoder checkpoint: "
            f"{args.latent_dim} vs {checkpoint_latent_dim}."
        )

    set_seed(args.seed, deterministic=args.deterministic)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    plot_dir = ensure_dir(output_dir / "plots")
    log_path = output_dir / "training_log.csv"
    history_path = output_dir / "history.json"

    bundle = prepare_datasets(
        data_dir=Path(args.data_dir),
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
    symmetric_range = False

    condition_range = tuple(autoencoder_config.get("condition_range", (0.0, 1.0, 0.0, 1.0)))

    electron_train_dataset = SingleDomainCalorimeterDataset(
        samples=bundle.electron_train,
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        domain_name="electron",
    )
    photon_train_dataset = SingleDomainCalorimeterDataset(
        samples=bundle.photon_train,
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        domain_name="photon",
    )

    electron_eval_source = bundle.electron_val if bundle.electron_val else bundle.electron_train
    photon_eval_source = bundle.photon_val if bundle.photon_val else bundle.photon_train
    electron_eval_dataset = SingleDomainCalorimeterDataset(
        samples=select_fixed_samples(electron_eval_source, int(args.sample_eval_count), int(args.seed)),
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        domain_name="electron",
    )
    photon_eval_dataset = SingleDomainCalorimeterDataset(
        samples=select_fixed_samples(photon_eval_source, int(args.sample_eval_count), int(args.seed) + 1),
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        domain_name="photon",
    )

    electron_train_loader = build_loader(electron_train_dataset, int(args.batch_size), True, int(args.num_workers), int(args.seed))
    photon_train_loader = build_loader(photon_train_dataset, int(args.batch_size), True, int(args.num_workers), int(args.seed) + 1)
    electron_eval_loader = build_loader(electron_eval_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 20)
    photon_eval_loader = build_loader(photon_eval_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 21)

    autoencoder = build_autoencoder_from_config(autoencoder_config).to(device)
    if "model" in autoencoder_checkpoint:
        autoencoder.load_state_dict(autoencoder_checkpoint["model"])
    else:
        autoencoder.encoder.load_state_dict(autoencoder_checkpoint["encoder"])
        autoencoder.decoder.load_state_dict(autoencoder_checkpoint["decoder"])
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    encoder.eval()
    decoder.eval()
    set_requires_grad([encoder, decoder], False)

    flow = ConditionalLatentFlow(
        latent_dim=int(args.latent_dim),
        hidden_dims=tuple(int(value) for value in args.flow_hidden_dims),
        num_blocks=int(args.flow_blocks),
        num_domains=2,
        domain_embedding_dim=int(args.domain_embedding_dim),
        scale_clamp=float(args.scale_clamp),
        use_actnorm=bool(args.use_actnorm),
    ).to(device)

    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )
    scheduler = build_linear_scheduler(optimizer, int(args.epochs), int(args.epochs_decay))

    config = serialise_config(args)
    config.pop("use_standardization", None)
    config["condition_range"] = [float(value) for value in condition_range]
    config["image_scaling"] = IMAGE_SCALING_MODE
    config["autoencoder_checkpoint"] = args.autoencoder_checkpoint
    config["autoencoder_output_dir"] = autoencoder_config.get("output_dir")
    save_json(config, output_dir / "config.json")

    history: list[dict[str, float | int]] = []
    best_metric = float("inf")
    start_epoch = 1

    if args.resume:
        latest_path = latest_checkpoint_path(checkpoint_dir, extension=".pt")
        if latest_path is None:
            print("Resume requested but no stage-2 checkpoint was found; starting from scratch.", flush=True)
        else:
            checkpoint = torch.load(latest_path, map_location=device)
            flow.load_state_dict(checkpoint["flow"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = list(checkpoint.get("history", []))
            best_metric = float(checkpoint.get("best_metric", best_metric))
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"Resumed flow training from {latest_path} at epoch {start_epoch}.", flush=True)

    steps_per_epoch = max(len(electron_train_loader), len(photon_train_loader))
    electron_iterator = infinite_loader(electron_train_loader)
    photon_iterator = infinite_loader(photon_train_loader)

    total_epochs = int(args.epochs) + int(args.epochs_decay)
    for epoch in range(start_epoch, total_epochs + 1):
        flow.train()
        running = {
            "loss": 0.0,
            "nll_electron": 0.0,
            "nll_photon": 0.0,
            "mmd": 0.0,
            "moment": 0.0,
        }

        for step in range(1, steps_per_epoch + 1):
            batch_e = next(electron_iterator)
            batch_p = next(photon_iterator)

            electron_images = batch_e["image"].to(device)
            photon_images = batch_p["image"].to(device)
            with torch.no_grad():
                z_electron = encoder(electron_images)
                z_photon = encoder(photon_images)

            optimizer.zero_grad(set_to_none=True)
            electron_nll = -flow.log_prob(z_electron, 0).mean()
            photon_nll = -flow.log_prob(z_photon, 1).mean()
            transported = flow.transport(z_electron, source_domain=0, target_domain=1)
            mmd_loss = maximum_mean_discrepancy(transported, z_photon)
            moment_loss = latent_moment_loss(transported, z_photon)
            loss = electron_nll + photon_nll + float(args.lambda_mmd) * mmd_loss + float(args.lambda_moment) * moment_loss
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()

            running["loss"] += float(loss.item())
            running["nll_electron"] += float(electron_nll.item())
            running["nll_photon"] += float(photon_nll.item())
            running["mmd"] += float(mmd_loss.item())
            running["moment"] += float(moment_loss.item())

            if step % 50 == 0 or step == steps_per_epoch:
                print(
                    f"[FLOW][Epoch {epoch:03d}/{total_epochs:03d}] "
                    f"[Batch {step:04d}/{steps_per_epoch:04d}] "
                    f"[loss={loss.item():.6f}] [nll_e={electron_nll.item():.6f}] "
                    f"[nll_p={photon_nll.item():.6f}] [mmd={mmd_loss.item():.6f}] "
                    f"[moment={moment_loss.item():.6f}]",
                    flush=True,
                )

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running["loss"] / max(steps_per_epoch, 1),
            "train_nll_electron": running["nll_electron"] / max(steps_per_epoch, 1),
            "train_nll_photon": running["nll_photon"] / max(steps_per_epoch, 1),
            "train_mmd": running["mmd"] / max(steps_per_epoch, 1),
            "train_moment": running["moment"] / max(steps_per_epoch, 1),
            "val_selection_score": np.nan,
            "val_latent_transport_mmd": np.nan,
            "val_total_energy_w1": np.nan,
        }

        validation_metrics = None
        if epoch % int(args.validation_interval) == 0 or epoch == total_epochs:
            validation_metrics = run_flow_validation(
                encoder,
                decoder,
                flow,
                electron_eval_loader,
                photon_eval_loader,
                device,
                plot_dir / f"epoch_{epoch:03d}",
                max_plot_items=int(args.sample_plot_count),
            )
            row["val_selection_score"] = float(validation_metrics["selection_score"])
            row["val_latent_transport_mmd"] = float(validation_metrics["latent_transport_mmd"])
            row["val_total_energy_w1"] = float(validation_metrics["translated_photon_vs_photon_total_energy_w1"])
            print(
                f"Flow validation plots written to {plot_dir / f'epoch_{epoch:03d}'} "
                f"(selection_score={validation_metrics['selection_score']:.6f}).",
                flush=True,
            )

        history.append(row)
        save_json(history, history_path)
        from flow.utils import append_csv_row  # local import keeps the top-level imports shorter

        append_csv_row(log_path, row)
        plot_training_history(
            history,
            output_dir / "training_curves.png",
            keys=[
                ("train_loss", "Train total loss"),
                ("train_nll_electron", "Electron NLL"),
                ("train_nll_photon", "Photon NLL"),
                ("train_mmd", "Latent transport MMD"),
                ("val_selection_score", "Validation selection score"),
            ],
            title="Latent flow training history",
        )

        is_best = validation_metrics is not None and float(validation_metrics["selection_score"]) < best_metric
        if is_best:
            best_metric = float(validation_metrics["selection_score"])

        checkpoint_state = {
            "epoch": epoch,
            "config": config,
            "flow": flow.state_dict(),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_metric": best_metric,
            "autoencoder_checkpoint": args.autoencoder_checkpoint,
        }
        torch.save(checkpoint_state, checkpoint_dir / "latest.pt")
        if epoch % int(args.save_interval) == 0 or epoch == total_epochs:
            torch.save(checkpoint_state, checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(checkpoint_state, checkpoint_dir / "best.pt")
            print(
                f"New best flow checkpoint saved to {checkpoint_dir / 'best.pt'} "
                f"(selection_score={best_metric:.6f}).",
                flush=True,
            )

    print(f"Flow training complete. Latest checkpoint: {checkpoint_dir / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
