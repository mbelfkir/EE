from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/infer.py requires torch and numpy. Install the packages listed in ML/requirements.txt."
    ) from exc

from dataset import SingleDomainCalorimeterDataset, get_split_samples, prepare_datasets
from data import restore_physical_array
from models import ResnetGenerator, VoxelGenerator
from utils import choose_device, ensure_dir, normalized_to_raw_numpy, save_tensor_png, save_window_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained electron/photon CycleGAN.")
    parser.add_argument("--data-dir", default=None, help="Directory containing the ROOT calorimeter files")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train.py")
    parser.add_argument("--results-dir", default="outputs/cyclegan/inference", help="Directory for translated outputs")
    parser.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-standardization", dest="use_standardization", action="store_true")
    parser.add_argument("--no-standardization", dest="use_standardization", action="store_false")
    parser.set_defaults(use_standardization=None)
    parser.add_argument("--min-window-energy-gev", type=float, default=None)
    parser.add_argument("--image-transform", choices=["none", "cbrt"], default=None)
    parser.add_argument("--eta-window-size", type=int, default=None)
    parser.add_argument("--phi-window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on the number of translated samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    data_dir = Path(args.data_dir if args.data_dir is not None else config["data_dir"]).resolve()
    train_fraction = float(args.train_fraction if args.train_fraction is not None else config["train_fraction"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    use_standardization = (
        bool(config.get("use_standardization", False))
        if args.use_standardization is None
        else bool(args.use_standardization)
    )
    min_window_energy_gev = float(
        args.min_window_energy_gev
        if args.min_window_energy_gev is not None
        else config.get("min_window_energy_gev", 0.0)
    )
    image_transform = str(
        args.image_transform
        if args.image_transform is not None
        else config.get("image_transform", "none")
    )
    eta_window_size = int(args.eta_window_size if args.eta_window_size is not None else config["eta_window_size"])
    phi_window_size = int(args.phi_window_size if args.phi_window_size is not None else config["phi_window_size"])
    model_backend = str(config.get("model_backend", "image"))
    condition_range = tuple(config.get("condition_range", (0.0, 1.0, 0.0, 1.0)))

    if model_backend == "voxel":
        generator = VoxelGenerator(
            num_voxels=phi_window_size * eta_window_size,
            hidden_dims=config.get("generator_hidden_dims", [100, 200, 400]),
            activation=str(config.get("generator_activation", "silu")),
        ).to(device)
    else:
        generator = ResnetGenerator(
            input_nc=1,
            output_nc=1,
            ngf=int(config["ngf"]),
            n_blocks=int(config["n_residual_blocks"]),
        ).to(device)
    state_key = "G_AB" if args.direction == "AtoB" else "G_BA"
    generator.load_state_dict(checkpoint[state_key])
    generator.eval()

    bundle = prepare_datasets(
        data_dir=data_dir,
        train_fraction=train_fraction,
        seed=seed,
        use_standardization=use_standardization,
        backend="cnn",
        eta_window_size=eta_window_size,
        phi_window_size=phi_window_size,
        min_window_energy_gev=min_window_energy_gev,
        image_transform=image_transform,
        representation="voxel" if model_backend == "voxel" else "image",
    )
    symmetric_range = bundle.standardization is not None
    electron_samples, photon_samples = get_split_samples(bundle, args.split)
    input_samples = electron_samples if args.direction == "AtoB" else photon_samples
    dataset = SingleDomainCalorimeterDataset(
        samples=input_samples,
        normalization_scale=float(config["normalization_scale"]),
        phi_window_size=phi_window_size,
        eta_window_size=eta_window_size,
        representation=model_backend,
        condition_range=condition_range,
        symmetric_range=symmetric_range,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    results_dir = ensure_dir(Path(args.results_dir) / args.direction)
    translated_dir = ensure_dir(results_dir / "translated")
    raw_dir = ensure_dir(results_dir / "translated_raw")
    input_dir = ensure_dir(results_dir / "inputs")

    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            conditions = batch["cond"].to(device)
            energy_scales = batch["energy_scale"].to(device)
            if model_backend == "voxel":
                fake_energy = generator(inputs, conditions)
            else:
                fake = generator(inputs)
            paths = batch["path"]
            for index in range(inputs.size(0)):
                source_path = Path(paths[index])
                stem = source_path.stem
                if model_backend == "voxel":
                    scale = float(energy_scales[index].item())
                    input_window = (
                        inputs[index, :, 0].reshape(phi_window_size, eta_window_size).detach().cpu().numpy().astype(np.float32)
                        * scale
                    )
                    translated_window = (
                        fake_energy[index].reshape(phi_window_size, eta_window_size).detach().cpu().numpy().astype(np.float32)
                        * scale
                    )
                    input_physical = restore_physical_array(input_window, preprocessing=bundle.preprocessing, standardization=bundle.standardization)
                    raw = restore_physical_array(translated_window, preprocessing=bundle.preprocessing, standardization=bundle.standardization)
                    save_window_png(input_physical, input_dir / f"{stem}_input.png")
                    save_window_png(raw, translated_dir / f"{stem}_translated.png")
                else:
                    save_tensor_png(inputs[index], input_dir / f"{stem}_input.png")
                    save_tensor_png(fake[index], translated_dir / f"{stem}_translated.png")
                    raw = normalized_to_raw_numpy(
                        fake[index],
                        float(config["normalization_scale"]),
                        symmetric_range=symmetric_range,
                    )
                    raw = restore_physical_array(raw, preprocessing=bundle.preprocessing, standardization=bundle.standardization)
                np.save(raw_dir / f"{stem}_translated.npy", raw)
                processed += 1
                if args.max_images is not None and processed >= args.max_images:
                    print(f"Saved {processed} translated samples to {results_dir}")
                    return

    print(f"Saved {processed} translated samples to {results_dir}")


if __name__ == "__main__":
    main()
