from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flow.config import build_autoencoder_parser, canonicalise_autoencoder_args
from flow.datasets import (
    IMAGE_SCALING_MODE,
    SingleDomainCalorimeterDataset,
    UnpairedCalorimeterDataset,
    prepare_datasets,
    save_dataset_manifest,
    select_fixed_samples,
)
from flow.losses import latent_alignment_loss, reconstruction_loss
from flow.models import build_autoencoder
from flow.plotting import plot_training_history
from flow.utils import (
    build_linear_scheduler,
    choose_device,
    ensure_dir,
    latest_checkpoint_path,
    make_torch_generator,
    save_json,
    seed_worker,
    serialise_config,
    set_seed,
)
from flow.validate import evaluate_autoencoder_loss, evaluate_latent_alignment, run_autoencoder_validation


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


def main() -> None:
    args = canonicalise_autoencoder_args(build_autoencoder_parser().parse_args())
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

    train_dataset = UnpairedCalorimeterDataset(
        electron_samples=bundle.electron_train,
        photon_samples=bundle.photon_train,
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        symmetric_range=symmetric_range,
        match_beam_energy=True,
        serial_batches=False,
    )
    condition_range = train_dataset.condition_range

    electron_val_dataset = SingleDomainCalorimeterDataset(
        samples=bundle.electron_val if bundle.electron_val else bundle.electron_train,
        normalization_scale=0.0,
        phi_window_size=int(args.phi_window_size),
        eta_window_size=int(args.eta_window_size),
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        domain_name="electron",
    )
    photon_val_dataset = SingleDomainCalorimeterDataset(
        samples=bundle.photon_val if bundle.photon_val else bundle.photon_train,
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

    train_loader = build_loader(train_dataset, int(args.batch_size), True, int(args.num_workers), int(args.seed))
    electron_val_loader = build_loader(electron_val_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 10)
    photon_val_loader = build_loader(photon_val_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 11)
    electron_eval_loader = build_loader(electron_eval_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 20)
    photon_eval_loader = build_loader(photon_eval_dataset, int(args.batch_size), False, int(args.num_workers), int(args.seed) + 21)

    model = build_autoencoder(
        image_shape=(1, int(args.phi_window_size), int(args.eta_window_size)),
        latent_dim=int(args.latent_dim),
        channels=tuple(int(value) for value in args.encoder_channels),
        norm=str(args.norm),
        activation=str(args.activation),
        output_activation=str(args.output_activation),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )
    scheduler = build_linear_scheduler(optimizer, int(args.epochs), int(args.epochs_decay))

    config = serialise_config(args)
    config.pop("use_standardization", None)
    config["condition_range"] = [float(value) for value in condition_range]
    config["image_scaling"] = IMAGE_SCALING_MODE
    config["dataset_counts"] = {
        "electron_train": len(bundle.electron_train),
        "electron_val": len(bundle.electron_val),
        "photon_train": len(bundle.photon_train),
        "photon_val": len(bundle.photon_val),
    }
    save_json(config, output_dir / "config.json")

    history: list[dict[str, float | int]] = []
    best_metric = float("inf")
    start_epoch = 1

    if args.resume:
        latest_path = latest_checkpoint_path(checkpoint_dir, extension=".pt")
        if latest_path is None:
            print("Resume requested but no checkpoint was found; starting from scratch.", flush=True)
        else:
            checkpoint = torch.load(latest_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = list(checkpoint.get("history", []))
            best_metric = float(checkpoint.get("best_metric", best_metric))
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"Resumed autoencoder training from {latest_path} at epoch {start_epoch}.", flush=True)

    total_epochs = int(args.epochs) + int(args.epochs_decay)
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_reco_loss = 0.0
        train_align_loss = 0.0
        train_l1 = 0.0
        train_l2 = 0.0
        train_image_count = 0
        train_pair_count = 0

        for step, batch in enumerate(train_loader, start=1):
            electron_images = batch["A"].to(device)
            photon_images = batch["B"].to(device)
            optimizer.zero_grad(set_to_none=True)

            electron_reconstructed, electron_latent = model.reconstruct(electron_images)
            photon_reconstructed, photon_latent = model.reconstruct(photon_images)
            electron_reco, electron_pieces = reconstruction_loss(
                electron_reconstructed,
                electron_images,
                l1_weight=float(args.l1_weight),
                l2_weight=float(args.l2_weight),
            )
            photon_reco, photon_pieces = reconstruction_loss(
                photon_reconstructed,
                photon_images,
                l1_weight=float(args.l1_weight),
                l2_weight=float(args.l2_weight),
            )
            reco_loss = 0.5 * (electron_reco + photon_reco)
            align_loss = latent_alignment_loss(
                electron_latent,
                photon_latent,
                mode=str(args.latent_align_loss),
            )
            total_loss = reco_loss + float(args.lambda_latent_align) * align_loss

            total_loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            electron_batch_size = electron_images.size(0)
            photon_batch_size = photon_images.size(0)
            pair_batch_size = min(electron_batch_size, photon_batch_size)
            total_image_batch = electron_batch_size + photon_batch_size

            train_total_loss += float(total_loss.item()) * pair_batch_size
            train_reco_loss += (
                float(electron_reco.item()) * electron_batch_size
                + float(photon_reco.item()) * photon_batch_size
            )
            train_align_loss += float(align_loss.item()) * pair_batch_size
            train_l1 += (
                float(electron_pieces["l1"].item()) * electron_batch_size
                + float(photon_pieces["l1"].item()) * photon_batch_size
            )
            train_l2 += (
                float(electron_pieces["l2"].item()) * electron_batch_size
                + float(photon_pieces["l2"].item()) * photon_batch_size
            )
            train_image_count += total_image_batch
            train_pair_count += pair_batch_size

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"[AE][Epoch {epoch:03d}/{total_epochs:03d}] "
                    f"[Batch {step:04d}/{len(train_loader):04d}] "
                    f"[total={total_loss.item():.6f}] [reco={reco_loss.item():.6f}] "
                    f"[align={align_loss.item():.6f}] [e_l1={electron_pieces['l1'].item():.6f}] "
                    f"[p_l1={photon_pieces['l1'].item():.6f}]",
                    flush=True,
                )

        scheduler.step()
        train_image_denominator = max(train_image_count, 1)
        train_pair_denominator = max(train_pair_count, 1)
        electron_val_metrics = evaluate_autoencoder_loss(
            model.encoder,
            model.decoder,
            electron_val_loader,
            device,
            l1_weight=float(args.l1_weight),
            l2_weight=float(args.l2_weight),
        )
        photon_val_metrics = evaluate_autoencoder_loss(
            model.encoder,
            model.decoder,
            photon_val_loader,
            device,
            l1_weight=float(args.l1_weight),
            l2_weight=float(args.l2_weight),
        )
        val_reco = 0.5 * (electron_val_metrics["loss"] + photon_val_metrics["loss"])
        latent_val_metrics = evaluate_latent_alignment(
            model.encoder,
            electron_eval_loader,
            photon_eval_loader,
            device,
            alignment_loss_name=str(args.latent_align_loss),
        )
        val_total = val_reco + float(args.lambda_latent_align) * float(latent_val_metrics["latent_align_loss"])
        selection_score = val_reco + float(args.selection_alpha) * float(latent_val_metrics["avg_latent_w1"])

        if epoch == 1 or epoch % int(args.validation_interval) == 0 or epoch == total_epochs:
            validation_metrics = run_autoencoder_validation(
                model.encoder,
                model.decoder,
                electron_eval_loader,
                photon_eval_loader,
                device,
                plot_dir / f"epoch_{epoch:03d}",
                max_plot_items=int(args.sample_plot_count),
            )
            print(
                f"Autoencoder validation plots written to {plot_dir / f'epoch_{epoch:03d}'} "
                f"(combined_scaled_l1={validation_metrics['combined_scaled_l1']:.6f}, "
                f"avg_latent_w1={validation_metrics['avg_latent_w1']:.6f}).",
                flush=True,
            )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total_loss": train_total_loss / train_pair_denominator,
            "train_reco_loss": train_reco_loss / train_image_denominator,
            "train_latent_align_loss": train_align_loss / train_pair_denominator,
            "train_l1": train_l1 / train_image_denominator,
            "train_l2": train_l2 / train_image_denominator,
            "val_total_loss": float(val_total),
            "val_reco_loss": float(val_reco),
            "val_latent_align_loss": float(latent_val_metrics["latent_align_loss"]),
            "val_avg_latent_w1": float(latent_val_metrics["avg_latent_w1"]),
            "val_max_latent_w1": float(latent_val_metrics["max_latent_w1"]),
            "val_top5_avg_latent_w1": float(latent_val_metrics["top5_avg_latent_w1"]),
            "val_selection_score": float(selection_score),
            "val_electron_loss": float(electron_val_metrics["loss"]),
            "val_electron_l1": float(electron_val_metrics["l1"]),
            "val_electron_l2": float(electron_val_metrics["l2"]),
            "val_photon_loss": float(photon_val_metrics["loss"]),
            "val_photon_l1": float(photon_val_metrics["l1"]),
            "val_photon_l2": float(photon_val_metrics["l2"]),
        }
        history.append(row)
        save_json(history, history_path)
        from flow.utils import append_csv_row  # local import to keep the module list tidy

        append_csv_row(log_path, row)
        plot_training_history(
            history,
            output_dir / "training_curves.png",
            keys=[
                ("train_total_loss", "Train total loss"),
                ("val_total_loss", "Validation total loss"),
                ("train_reco_loss", "Train reconstruction loss"),
                ("train_latent_align_loss", "Train latent alignment loss"),
                ("val_avg_latent_w1", "Validation avg latent W1"),
                ("val_selection_score", "Stage-1 selection score"),
            ],
            title="Autoencoder training history with latent alignment",
        )

        is_best = selection_score < best_metric
        if is_best:
            best_metric = float(selection_score)

        checkpoint_state = {
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_metric": best_metric,
            "best_metric_name": "stage1_selection_score",
        }
        torch.save(checkpoint_state, checkpoint_dir / "latest.pt")
        if epoch % int(args.save_interval) == 0 or epoch == total_epochs:
            torch.save(checkpoint_state, checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(checkpoint_state, checkpoint_dir / "best.pt")
            print(
                f"New best autoencoder checkpoint saved to {checkpoint_dir / 'best.pt'} "
                f"(selection_score={best_metric:.6f}, val_reco={val_reco:.6f}, "
                f"avg_latent_w1={latent_val_metrics['avg_latent_w1']:.6f}).",
                flush=True,
            )

    print(f"Autoencoder training complete. Latest checkpoint: {checkpoint_dir / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
