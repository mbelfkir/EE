from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/train.py requires torch. Install the packages listed in ML/requirements.txt."
    ) from exc

from dataset import (
    DEFAULT_IMAGE_TRANSFORM,
    DEFAULT_ETA_WINDOW_SIZE,
    DEFAULT_MIN_WINDOW_ENERGY_GEV,
    DEFAULT_PHI_WINDOW_SIZE,
    DEFAULT_USE_STANDARDIZATION,
    UnpairedCalorimeterDataset,
    compute_normalization_scale,
    normalize_condition,
    particle_energy_scale_mev,
    prepare_datasets,
    save_dataset_manifest,
)
from models import PatchGANDiscriminator, ResnetGenerator, VoxelCritic, VoxelGenerator, replace_voxel_energy
from sample_evaluation import run_sample_evaluation
from utils import (
    ReplayBuffer,
    append_csv_row,
    build_linear_scheduler,
    choose_device,
    ensure_dir,
    init_weights,
    latest_checkpoint_path,
    normalized_to_raw_tensor,
    save_json,
    save_tensor_png,
    set_requires_grad,
    set_seed,
)

IMAGE_DEFAULT_LAMBDA_CYCLE = 10.0
IMAGE_DEFAULT_LAMBDA_IDENTITY = 0.5
IMAGE_DEFAULT_LAMBDA_ENERGY = 1.0
IMAGE_DEFAULT_LAMBDA_GP = 10.0

VOXEL_DEFAULT_LAMBDA_CYCLE = 2.0
VOXEL_DEFAULT_LAMBDA_IDENTITY = 0.1
VOXEL_DEFAULT_LAMBDA_GP = 5.0
VOXEL_DEFAULT_CRITIC_UPDATES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN for electron <-> photon calorimeter translation.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the ROOT calorimeter files")
    parser.add_argument("--output-dir", default="outputs/cyclegan", help="Directory for checkpoints and logs")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--use-standardization", dest="use_standardization", action="store_true")
    parser.add_argument("--no-standardization", dest="use_standardization", action="store_false")
    parser.set_defaults(use_standardization=DEFAULT_USE_STANDARDIZATION)
    parser.add_argument("--min-window-energy-gev", type=float, default=DEFAULT_MIN_WINDOW_ENERGY_GEV)
    parser.add_argument("--image-transform", choices=["none", "cbrt"], default=DEFAULT_IMAGE_TRANSFORM)
    parser.add_argument("--eta-window-size", type=int, default=DEFAULT_ETA_WINDOW_SIZE)
    parser.add_argument("--phi-window-size", type=int, default=DEFAULT_PHI_WINDOW_SIZE)
    parser.add_argument("--model-backend", choices=["image", "voxel"], default="image")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs at constant learning rate")
    parser.add_argument("--n-epochs-decay", type=int, default=100, help="Number of linearly decaying epochs")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-cycle-a", type=float, default=IMAGE_DEFAULT_LAMBDA_CYCLE)
    parser.add_argument("--lambda-cycle-b", type=float, default=IMAGE_DEFAULT_LAMBDA_CYCLE)
    parser.add_argument("--lambda-identity", type=float, default=IMAGE_DEFAULT_LAMBDA_IDENTITY)
    parser.add_argument("--lambda-energy", type=float, default=IMAGE_DEFAULT_LAMBDA_ENERGY)
    parser.add_argument("--lambda-reta", type=float, default=1.0)
    parser.add_argument("--lambda-rphi", type=float, default=1.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n-residual-blocks", type=int, default=6)
    parser.add_argument("--disc-layers", type=int, default=3)
    parser.add_argument("--generator-hidden-dims", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[400, 200, 100])
    parser.add_argument("--generator-activation", choices=["relu", "silu", "swish", "lrelu"], default="silu")
    parser.add_argument("--lambda-gp", type=float, default=IMAGE_DEFAULT_LAMBDA_GP)
    parser.add_argument("--critic-updates", type=int, default=1, help="Number of critic updates per generator update")
    parser.add_argument("--normalization-scale", type=float, default=None)
    parser.add_argument("--serial-batches", action="store_true", help="Use deterministic A/B pairing by index")
    parser.add_argument("--sample-interval", type=int, default=5, help="Save translation examples every N epochs")
    parser.add_argument(
        "--sample-eval-count",
        type=int,
        default=1024,
        help="Number of fixed validation samples per domain used for sample-stage comparison plots; <=0 uses all",
    )
    parser.add_argument(
        "--sample-eval-batch-size",
        type=int,
        default=64,
        help="Batch size for sample-stage translation and shower-shape evaluation",
    )
    parser.add_argument("--save-interval", type=int, default=5, help="Save epoch checkpoints every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output-dir/checkpoints")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def specified_cli_flags(argv: Sequence[str] | None = None) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    flags: set[str] = set()
    for token in tokens:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def gan_loss(prediction: Tensor, target_is_real: bool, criterion: nn.Module) -> Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return criterion(prediction, target)


def restore_physical_tensor(
    tensor: Tensor,
    preprocessing: dict[str, object] | None,
    standardization: dict[str, object] | None,
) -> Tensor:
    restored = tensor
    if standardization and standardization.get("mode") == "global":
        mean = float(standardization["mean"])
        std = float(standardization["std"])
        restored = restored * std + mean
    image_transform = str(preprocessing.get("image_transform", "none")) if preprocessing else "none"
    if image_transform == "cbrt":
        restored = torch.clamp(restored, min=0.0).pow(3.0)
    elif image_transform != "none":
        raise ValueError(f"Unsupported image_transform: {image_transform}")
    return restored


def energy_conservation_loss(
    fake: Tensor,
    real: Tensor,
    normalization_scale: float,
    symmetric_range: bool,
    preprocessing: dict[str, object] | None,
    standardization: dict[str, object] | None,
) -> Tensor:
    fake_energy = normalized_to_raw_tensor(fake, normalization_scale, symmetric_range=symmetric_range)
    real_energy = normalized_to_raw_tensor(real, normalization_scale, symmetric_range=symmetric_range)
    if standardization and standardization.get("mode") == "global":
        mean = float(standardization["mean"])
        std = float(standardization["std"])
        fake_energy = fake_energy * std + mean
        real_energy = real_energy * std + mean
    image_transform = str(preprocessing.get("image_transform", "none")) if preprocessing else "none"
    if image_transform == "cbrt":
        fake_energy = torch.clamp(fake_energy, min=0.0).pow(3.0)
        real_energy = torch.clamp(real_energy, min=0.0).pow(3.0)
    elif image_transform != "none":
        raise ValueError(f"Unsupported image_transform: {image_transform}")
    fake_energy = fake_energy.sum(dim=(1, 2, 3))
    real_energy = real_energy.sum(dim=(1, 2, 3))
    return torch.mean(torch.abs(fake_energy - real_energy))


def voxel_energy_conservation_loss(
    fake_energy: Tensor,
    real_voxel: Tensor,
    fake_energy_scale: Tensor,
    real_energy_scale: Tensor,
) -> Tensor:
    fake_physical = fake_energy * fake_energy_scale.reshape(-1, 1)
    real_physical = real_voxel[:, :, 0] * real_energy_scale.reshape(-1, 1)
    return torch.mean(torch.abs(fake_physical.sum(dim=1) - real_physical.sum(dim=1)))


def voxel_energy_to_images(energy: Tensor, phi_window_size: int, eta_window_size: int) -> Tensor:
    return energy.view(energy.size(0), 1, phi_window_size, eta_window_size)


def samples_to_conditions(samples: Sequence, condition_range: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(
        [normalize_condition(sample, condition_range) for sample in samples],
        dtype=np.float32,
    )


def samples_to_energy_scales(samples: Sequence) -> np.ndarray:
    return np.asarray([particle_energy_scale_mev(sample) for sample in samples], dtype=np.float32)[:, None]


def gradient_penalty(
    critic: nn.Module,
    real_voxel: Tensor,
    fake_voxel: Tensor,
    condition: Tensor,
) -> Tensor:
    batch_size = real_voxel.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_voxel.device, dtype=real_voxel.dtype)
    interpolated = alpha * real_voxel + (1.0 - alpha) * fake_voxel.detach()
    interpolated.requires_grad_(True)
    scores = critic(interpolated, condition).view(-1)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def torch_sum_window(energy_images: Tensor, eta_bins: int, phi_bins: int) -> Tensor:
    center_eta = energy_images.size(2) // 2
    center_phi = energy_images.size(1) // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_slice = slice(center_eta - eta_half, center_eta + eta_half + 1)
    phi_slice = slice(center_phi - phi_half, center_phi + phi_half + 1)
    return energy_images[:, phi_slice, eta_slice].sum(dim=(1, 2))


def torch_reta_rphi(fake_energy: Tensor, phi_window_size: int, eta_window_size: int) -> Tensor:
    energy_images = fake_energy.reshape(-1, phi_window_size, eta_window_size)
    e7x7 = torch_sum_window(energy_images, 7, 7)
    e3x7 = torch_sum_window(energy_images, 3, 7)
    e3x3 = torch_sum_window(energy_images, 3, 3)
    zero = torch.zeros_like(e7x7)
    reta = torch.where(e7x7 > 0.0, e3x7 / e7x7, zero)
    rphi = torch.where(e3x7 > 0.0, e3x3 / e3x7, zero)
    return torch.stack([reta, rphi], dim=1)


def build_model_description(args: argparse.Namespace, num_voxels: int) -> str:
    if args.model_backend == "voxel":
        return (
            "Voxel CycleGAN backend with paper-inspired dense conditional generator/critic "
            f"(num_voxels={num_voxels}, voxel_features=5, generator_hidden_dims={list(args.generator_hidden_dims)}, "
            f"critic_hidden_dims={list(args.critic_hidden_dims)}, generator_activation={args.generator_activation}, "
            "critic=spectral_norm_mlp, adversarial_loss=wgan_gp, voxel_energy_normalization=particle_energy, "
            f"condition=reta_rphi, explicit_energy_loss=off, lambda_cycle={args.lambda_cycle_a}/{args.lambda_cycle_b}, "
            f"lambda_identity={args.lambda_identity}, lambda_reta={args.lambda_reta}, "
            f"lambda_rphi={args.lambda_rphi}, lambda_gp={args.lambda_gp}, "
            f"critic_updates={args.critic_updates})."
        )
    return (
        "Image CycleGAN backend with ResNet generator and PatchGAN discriminator "
        f"(ngf={args.ngf}, ndf={args.ndf}, n_residual_blocks={args.n_residual_blocks}, disc_layers={args.disc_layers})."
    )


def save_samples(
    generator_ab: nn.Module,
    generator_ba: nn.Module,
    batch: dict[str, Tensor | str],
    output_dir: Path,
    epoch: int,
) -> None:
    generator_ab.eval()
    generator_ba.eval()
    sample_dir = ensure_dir(output_dir / "samples" / f"epoch_{epoch:04d}")
    device = next(generator_ab.parameters()).device
    with torch.no_grad():
        real_a = batch["A"].to(device)
        real_b = batch["B"].to(device)
        fake_b = generator_ab(real_a)
        fake_a = generator_ba(real_b)
        rec_a = generator_ba(fake_b)
        rec_b = generator_ab(fake_a)

    for index in range(min(real_a.size(0), 4)):
        save_tensor_png(real_a[index], sample_dir / f"{index:02d}_real_A.png")
        save_tensor_png(fake_b[index], sample_dir / f"{index:02d}_fake_B.png")
        save_tensor_png(rec_a[index], sample_dir / f"{index:02d}_rec_A.png")
        save_tensor_png(real_b[index], sample_dir / f"{index:02d}_real_B.png")
        save_tensor_png(fake_a[index], sample_dir / f"{index:02d}_fake_A.png")
        save_tensor_png(rec_b[index], sample_dir / f"{index:02d}_rec_B.png")

    generator_ab.train()
    generator_ba.train()


def select_fixed_samples(samples: Sequence, max_count: int, seed: int) -> list:
    samples = list(samples)
    if max_count <= 0 or len(samples) <= max_count:
        return samples
    indices = sorted(random.Random(seed).sample(range(len(samples)), max_count))
    return [samples[index] for index in indices]


def samples_to_array(samples: Sequence) -> np.ndarray:
    return np.stack([sample.image for sample in samples], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    cli_flags = specified_cli_flags()
    args.data_dir = str(Path(args.data_dir).resolve())
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_path = output_dir / "training_log.csv"

    start_epoch = 1
    latest_path = latest_checkpoint_path(checkpoint_dir) if args.resume else None
    checkpoint = None
    checkpoint_config: dict[str, object] | None = None
    if latest_path is not None:
        checkpoint = torch.load(latest_path, map_location=device)
        checkpoint_config = dict(checkpoint["config"])
        if args.normalization_scale is None:
            args.normalization_scale = float(checkpoint_config["normalization_scale"])
        args.use_standardization = bool(checkpoint_config.get("use_standardization", False))
        args.min_window_energy_gev = float(checkpoint_config.get("min_window_energy_gev", 0.0))
        args.image_transform = str(checkpoint_config.get("image_transform", "none"))
        args.model_backend = str(checkpoint_config.get("model_backend", "image"))
        args.generator_hidden_dims = [int(value) for value in checkpoint_config.get("generator_hidden_dims", args.generator_hidden_dims)]
        args.critic_hidden_dims = [int(value) for value in checkpoint_config.get("critic_hidden_dims", args.critic_hidden_dims)]
        args.generator_activation = str(checkpoint_config.get("generator_activation", args.generator_activation))
        args.lambda_gp = float(checkpoint_config.get("lambda_gp", args.lambda_gp))
        args.critic_updates = int(checkpoint_config.get("critic_updates", args.critic_updates))
        start_epoch = int(checkpoint["epoch"]) + 1

    if args.model_backend == "voxel":
        if checkpoint_config is None:
            if "--lambda-cycle-a" not in cli_flags and args.lambda_cycle_a == IMAGE_DEFAULT_LAMBDA_CYCLE:
                args.lambda_cycle_a = VOXEL_DEFAULT_LAMBDA_CYCLE
            if "--lambda-cycle-b" not in cli_flags and args.lambda_cycle_b == IMAGE_DEFAULT_LAMBDA_CYCLE:
                args.lambda_cycle_b = VOXEL_DEFAULT_LAMBDA_CYCLE
            if "--lambda-identity" not in cli_flags and args.lambda_identity == IMAGE_DEFAULT_LAMBDA_IDENTITY:
                args.lambda_identity = VOXEL_DEFAULT_LAMBDA_IDENTITY
            if "--lambda-gp" not in cli_flags and args.lambda_gp == IMAGE_DEFAULT_LAMBDA_GP:
                args.lambda_gp = VOXEL_DEFAULT_LAMBDA_GP
            if "--critic-updates" not in cli_flags and args.critic_updates == 1:
                args.critic_updates = VOXEL_DEFAULT_CRITIC_UPDATES
        if args.use_standardization:
            print("Voxel backend uses particle-energy normalization; disabling global standardization.", flush=True)
        args.use_standardization = False
        if args.lambda_energy != 0.0:
            print("Voxel backend disables explicit energy loss; forcing lambda_energy=0.", flush=True)
        args.lambda_energy = 0.0
        print(
            "Voxel loss weights: "
            f"lambda_cycle_a={args.lambda_cycle_a}, "
            f"lambda_cycle_b={args.lambda_cycle_b}, "
            f"lambda_identity={args.lambda_identity}, "
            f"lambda_reta={args.lambda_reta}, "
            f"lambda_rphi={args.lambda_rphi}, "
            f"lambda_gp={args.lambda_gp}, "
            f"critic_updates={args.critic_updates}, "
            f"lambda_energy={args.lambda_energy}",
            flush=True,
        )

    print(
        f"Preparing datasets from {args.data_dir} with windows "
        f"{args.phi_window_size}x{args.eta_window_size} and train_fraction={args.train_fraction}",
        flush=True,
    )
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
        representation="voxel" if args.model_backend == "voxel" else "image",
    )
    print(
        "Loaded samples: "
        f"electron_train={len(bundle.electron_train)}, "
        f"electron_val={len(bundle.electron_val)}, "
        f"photon_train={len(bundle.photon_train)}, "
        f"photon_val={len(bundle.photon_val)}",
        flush=True,
    )
    save_dataset_manifest(bundle, output_dir)
    symmetric_range = bundle.standardization is not None

    if args.model_backend == "image" and args.normalization_scale is None:
        args.normalization_scale = compute_normalization_scale(
            [bundle.electron_train, bundle.photon_train],
            symmetric_range=symmetric_range,
        )
    elif args.model_backend == "voxel":
        args.normalization_scale = 1.0

    condition_range = (
        tuple(checkpoint_config.get("condition_range", (0.0, 1.0, 0.0, 1.0)))
        if checkpoint_config is not None and "condition_range" in checkpoint_config
        else None
    )

    train_dataset = UnpairedCalorimeterDataset(
        electron_samples=bundle.electron_train,
        photon_samples=bundle.photon_train,
        normalization_scale=args.normalization_scale,
        phi_window_size=args.phi_window_size,
        eta_window_size=args.eta_window_size,
        representation=args.model_backend,
        condition_range=condition_range,
        symmetric_range=symmetric_range,
        serial_batches=args.serial_batches,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.serial_batches,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Training loader ready: {len(train_loader)} batches per epoch", flush=True)

    electron_eval_source = bundle.electron_val if bundle.electron_val else bundle.electron_train
    photon_eval_source = bundle.photon_val if bundle.photon_val else bundle.photon_train
    eval_split = "val" if bundle.electron_val and bundle.photon_val else "train"
    electron_eval_samples = select_fixed_samples(electron_eval_source, int(args.sample_eval_count), int(args.seed))
    photon_eval_samples = select_fixed_samples(photon_eval_source, int(args.sample_eval_count), int(args.seed) + 1)
    electron_eval_windows = samples_to_array(electron_eval_samples)
    photon_eval_windows = samples_to_array(photon_eval_samples)
    eval_condition_range = train_dataset.condition_range
    electron_eval_conditions = samples_to_conditions(electron_eval_samples, eval_condition_range)
    photon_eval_conditions = samples_to_conditions(photon_eval_samples, eval_condition_range)
    electron_eval_energy_scales = samples_to_energy_scales(electron_eval_samples)
    photon_eval_energy_scales = samples_to_energy_scales(photon_eval_samples)
    print(
        f"Sample-stage comparison will use split={eval_split}, "
        f"electron={len(electron_eval_windows)}, photon={len(photon_eval_windows)}",
        flush=True,
    )
    num_voxels = int(args.phi_window_size) * int(args.eta_window_size)

    if args.model_backend == "voxel":
        generator_ab = VoxelGenerator(
            num_voxels=num_voxels,
            hidden_dims=args.generator_hidden_dims,
            activation=args.generator_activation,
        ).to(device)
        generator_ba = VoxelGenerator(
            num_voxels=num_voxels,
            hidden_dims=args.generator_hidden_dims,
            activation=args.generator_activation,
        ).to(device)
        discriminator_a = VoxelCritic(num_voxels=num_voxels, hidden_dims=args.critic_hidden_dims).to(device)
        discriminator_b = VoxelCritic(num_voxels=num_voxels, hidden_dims=args.critic_hidden_dims).to(device)
    else:
        generator_ab = ResnetGenerator(input_nc=1, output_nc=1, ngf=args.ngf, n_blocks=args.n_residual_blocks).to(device)
        generator_ba = ResnetGenerator(input_nc=1, output_nc=1, ngf=args.ngf, n_blocks=args.n_residual_blocks).to(device)
        discriminator_a = PatchGANDiscriminator(input_nc=1, ndf=args.ndf, n_layers=args.disc_layers).to(device)
        discriminator_b = PatchGANDiscriminator(input_nc=1, ndf=args.ndf, n_layers=args.disc_layers).to(device)

    if checkpoint is None:
        generator_ab.apply(init_weights)
        generator_ba.apply(init_weights)
        discriminator_a.apply(init_weights)
        discriminator_b.apply(init_weights)
    else:
        generator_ab.load_state_dict(checkpoint["G_AB"])
        generator_ba.load_state_dict(checkpoint["G_BA"])
        discriminator_a.load_state_dict(checkpoint["D_A"])
        discriminator_b.load_state_dict(checkpoint["D_B"])

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_g = torch.optim.Adam(
        itertools.chain(generator_ab.parameters(), generator_ba.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_d_a = torch.optim.Adam(discriminator_a.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d_b = torch.optim.Adam(discriminator_b.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    scheduler_g = build_linear_scheduler(optimizer_g, args.n_epochs, args.n_epochs_decay)
    scheduler_d_a = build_linear_scheduler(optimizer_d_a, args.n_epochs, args.n_epochs_decay)
    scheduler_d_b = build_linear_scheduler(optimizer_d_b, args.n_epochs, args.n_epochs_decay)

    if checkpoint is not None:
        optimizer_g.load_state_dict(checkpoint["optimizer_G"])
        optimizer_d_a.load_state_dict(checkpoint["optimizer_D_A"])
        optimizer_d_b.load_state_dict(checkpoint["optimizer_D_B"])
        scheduler_g.load_state_dict(checkpoint["scheduler_G"])
        scheduler_d_a.load_state_dict(checkpoint["scheduler_D_A"])
        scheduler_d_b.load_state_dict(checkpoint["scheduler_D_B"])

    fake_a_buffer = ReplayBuffer() if args.model_backend == "image" else None
    fake_b_buffer = ReplayBuffer() if args.model_backend == "image" else None

    config = vars(args).copy()
    config["condition_range"] = [float(value) for value in eval_condition_range]
    config["symmetric_normalization"] = symmetric_range
    config["model_description"] = build_model_description(args, num_voxels=num_voxels)
    save_json(config, output_dir / "config.json")

    total_epochs = args.n_epochs + args.n_epochs_decay
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_metrics = {
            "loss_G": 0.0,
            "loss_G_GAN": 0.0,
            "loss_G_cycle": 0.0,
            "loss_G_identity": 0.0,
            "loss_G_shape": 0.0,
            "loss_G_reta": 0.0,
            "loss_G_rphi": 0.0,
            "loss_G_energy": 0.0,
            "loss_D_A": 0.0,
            "loss_D_B": 0.0,
        }
        gen_updates = 0
        first_batch = None
        last_loss_g = torch.zeros((), device=device)
        last_loss_g_gan = torch.zeros((), device=device)
        last_loss_g_cycle = torch.zeros((), device=device)
        last_loss_g_identity = torch.zeros((), device=device)
        last_loss_g_shape = torch.zeros((), device=device)
        last_loss_g_reta = torch.zeros((), device=device)
        last_loss_g_rphi = torch.zeros((), device=device)
        last_loss_energy = torch.zeros((), device=device)

        for step, batch in enumerate(train_loader, start=1):
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)
            cond_a = batch["A_cond"].to(device)
            cond_b = batch["B_cond"].to(device)
            energy_scale_a = batch["A_energy_scale"].to(device)
            energy_scale_b = batch["B_energy_scale"].to(device)
            if first_batch is None:
                first_batch = {
                    "A": real_a[:4].detach().cpu(),
                    "B": real_b[:4].detach().cpu(),
                    "A_cond": cond_a[:4].detach().cpu(),
                    "B_cond": cond_b[:4].detach().cpu(),
                    "A_energy_scale": energy_scale_a[:4].detach().cpu(),
                    "B_energy_scale": energy_scale_b[:4].detach().cpu(),
                }

            if args.model_backend == "voxel":
                fake_b_energy = generator_ab(real_a, cond_a)
                fake_a_energy = generator_ba(real_b, cond_b)
                fake_b = replace_voxel_energy(real_a, fake_b_energy)
                fake_a = replace_voxel_energy(real_b, fake_a_energy)
                set_requires_grad([discriminator_a, discriminator_b], True)

                optimizer_d_a.zero_grad()
                gp_a = gradient_penalty(discriminator_a, real_a, fake_a, cond_b)
                loss_d_a = discriminator_a(fake_a.detach(), cond_b).mean() - discriminator_a(real_a, cond_a).mean() + args.lambda_gp * gp_a
                loss_d_a.backward()
                optimizer_d_a.step()

                optimizer_d_b.zero_grad()
                gp_b = gradient_penalty(discriminator_b, real_b, fake_b, cond_a)
                loss_d_b = discriminator_b(fake_b.detach(), cond_a).mean() - discriminator_b(real_b, cond_b).mean() + args.lambda_gp * gp_b
                loss_d_b.backward()
                optimizer_d_b.step()

                update_generator = step % args.critic_updates == 0 or step == len(train_loader)
                if update_generator:
                    set_requires_grad([discriminator_a, discriminator_b], False)
                    optimizer_g.zero_grad()

                    idt_a_energy = generator_ba(real_a, cond_a)
                    idt_b_energy = generator_ab(real_b, cond_b)
                    loss_idt_a = criterion_identity(idt_a_energy, real_a[:, :, 0]) * args.lambda_cycle_a * args.lambda_identity
                    loss_idt_b = criterion_identity(idt_b_energy, real_b[:, :, 0]) * args.lambda_cycle_b * args.lambda_identity

                    fake_b_energy = generator_ab(real_a, cond_a)
                    fake_a_energy = generator_ba(real_b, cond_b)
                    fake_b = replace_voxel_energy(real_a, fake_b_energy)
                    fake_a = replace_voxel_energy(real_b, fake_a_energy)
                    loss_gan_ab = -discriminator_b(fake_b, cond_a).mean()
                    loss_gan_ba = -discriminator_a(fake_a, cond_b).mean()

                    recovered_a_energy = generator_ba(fake_b, cond_a)
                    recovered_b_energy = generator_ab(fake_a, cond_b)
                    loss_cycle_a = criterion_cycle(recovered_a_energy, real_a[:, :, 0]) * args.lambda_cycle_a
                    loss_cycle_b = criterion_cycle(recovered_b_energy, real_b[:, :, 0]) * args.lambda_cycle_b

                    fake_shape_ab = torch_reta_rphi(fake_b_energy, int(args.phi_window_size), int(args.eta_window_size))
                    fake_shape_ba = torch_reta_rphi(fake_a_energy, int(args.phi_window_size), int(args.eta_window_size))
                    loss_shape_reta = criterion_identity(fake_shape_ab[:, 0], cond_a[:, 0]) + criterion_identity(fake_shape_ba[:, 0], cond_b[:, 0])
                    loss_shape_rphi = criterion_identity(fake_shape_ab[:, 1], cond_a[:, 1]) + criterion_identity(fake_shape_ba[:, 1], cond_b[:, 1])
                    loss_g_shape = args.lambda_reta * loss_shape_reta + args.lambda_rphi * loss_shape_rphi

                    loss_energy = torch.zeros((), device=device, dtype=real_a.dtype)

                    loss_g_identity = loss_idt_a + loss_idt_b
                    loss_g_gan = loss_gan_ab + loss_gan_ba
                    loss_g_cycle = loss_cycle_a + loss_cycle_b
                    loss_g = loss_g_identity + loss_g_gan + loss_g_cycle + loss_g_shape + loss_energy
                    loss_g.backward()
                    optimizer_g.step()

                    last_loss_g = loss_g.detach()
                    last_loss_g_gan = loss_g_gan.detach()
                    last_loss_g_cycle = loss_g_cycle.detach()
                    last_loss_g_identity = loss_g_identity.detach()
                    last_loss_g_shape = loss_g_shape.detach()
                    last_loss_g_reta = loss_shape_reta.detach()
                    last_loss_g_rphi = loss_shape_rphi.detach()
                    last_loss_energy = loss_energy.detach()

                    epoch_metrics["loss_G"] += float(last_loss_g.cpu())
                    epoch_metrics["loss_G_GAN"] += float(last_loss_g_gan.cpu())
                    epoch_metrics["loss_G_cycle"] += float(last_loss_g_cycle.cpu())
                    epoch_metrics["loss_G_identity"] += float(last_loss_g_identity.cpu())
                    epoch_metrics["loss_G_shape"] += float(last_loss_g_shape.cpu())
                    epoch_metrics["loss_G_reta"] += float(last_loss_g_reta.cpu())
                    epoch_metrics["loss_G_rphi"] += float(last_loss_g_rphi.cpu())
                    epoch_metrics["loss_G_energy"] += float(last_loss_energy.cpu())
                    gen_updates += 1
            else:
                set_requires_grad([discriminator_a, discriminator_b], False)
                optimizer_g.zero_grad()
                idt_a = generator_ba(real_a)
                idt_b = generator_ab(real_b)
                loss_idt_a = criterion_identity(idt_a, real_a) * args.lambda_cycle_a * args.lambda_identity
                loss_idt_b = criterion_identity(idt_b, real_b) * args.lambda_cycle_b * args.lambda_identity

                fake_b = generator_ab(real_a)
                fake_a = generator_ba(real_b)
                loss_gan_ab = gan_loss(discriminator_b(fake_b), True, criterion_gan)
                loss_gan_ba = gan_loss(discriminator_a(fake_a), True, criterion_gan)

                recovered_a = generator_ba(fake_b)
                recovered_b = generator_ab(fake_a)
                loss_cycle_a = criterion_cycle(recovered_a, real_a) * args.lambda_cycle_a
                loss_cycle_b = criterion_cycle(recovered_b, real_b) * args.lambda_cycle_b

                loss_energy_ab = energy_conservation_loss(
                    fake_b,
                    real_a,
                    normalization_scale=float(args.normalization_scale),
                    symmetric_range=symmetric_range,
                    preprocessing=bundle.preprocessing,
                    standardization=bundle.standardization,
                )
                loss_energy_ba = energy_conservation_loss(
                    fake_a,
                    real_b,
                    normalization_scale=float(args.normalization_scale),
                    symmetric_range=symmetric_range,
                    preprocessing=bundle.preprocessing,
                    standardization=bundle.standardization,
                )
                loss_energy = (loss_energy_ab + loss_energy_ba) * args.lambda_energy

                loss_g_identity = loss_idt_a + loss_idt_b
                loss_g_gan = loss_gan_ab + loss_gan_ba
                loss_g_cycle = loss_cycle_a + loss_cycle_b
                last_loss_g_shape = torch.zeros((), device=device, dtype=real_a.dtype)
                loss_g = loss_g_identity + loss_g_gan + loss_g_cycle + loss_energy
                loss_g.backward()
                optimizer_g.step()
                last_loss_g = loss_g.detach()
                last_loss_g_gan = loss_g_gan.detach()
                last_loss_g_cycle = loss_g_cycle.detach()
                last_loss_g_identity = loss_g_identity.detach()
                last_loss_g_reta = torch.zeros((), device=device, dtype=real_a.dtype)
                last_loss_g_rphi = torch.zeros((), device=device, dtype=real_a.dtype)
                last_loss_energy = loss_energy.detach()

                set_requires_grad([discriminator_a, discriminator_b], True)

                optimizer_d_a.zero_grad()
                buffered_fake_a = fake_a_buffer.push_and_pop(fake_a)
                loss_d_a_real = gan_loss(discriminator_a(real_a), True, criterion_gan)
                loss_d_a_fake = gan_loss(discriminator_a(buffered_fake_a.detach()), False, criterion_gan)
                loss_d_a = 0.5 * (loss_d_a_real + loss_d_a_fake)
                loss_d_a.backward()
                optimizer_d_a.step()

                optimizer_d_b.zero_grad()
                buffered_fake_b = fake_b_buffer.push_and_pop(fake_b)
                loss_d_b_real = gan_loss(discriminator_b(real_b), True, criterion_gan)
                loss_d_b_fake = gan_loss(discriminator_b(buffered_fake_b.detach()), False, criterion_gan)
                loss_d_b = 0.5 * (loss_d_b_real + loss_d_b_fake)
                loss_d_b.backward()
                optimizer_d_b.step()
                epoch_metrics["loss_G"] += float(last_loss_g.cpu())
                epoch_metrics["loss_G_GAN"] += float(last_loss_g_gan.cpu())
                epoch_metrics["loss_G_cycle"] += float(last_loss_g_cycle.cpu())
                epoch_metrics["loss_G_identity"] += float(last_loss_g_identity.cpu())
                epoch_metrics["loss_G_shape"] += float(last_loss_g_shape.cpu())
                epoch_metrics["loss_G_reta"] += float(last_loss_g_reta.cpu())
                epoch_metrics["loss_G_rphi"] += float(last_loss_g_rphi.cpu())
                epoch_metrics["loss_G_energy"] += float(last_loss_energy.cpu())
                gen_updates += 1
            epoch_metrics["loss_D_A"] += float(loss_d_a.detach().cpu())
            epoch_metrics["loss_D_B"] += float(loss_d_b.detach().cpu())

            if step % 50 == 0 or step == len(train_loader):
                if args.model_backend == "voxel":
                    print(
                        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
                        f"[Batch {step:04d}/{len(train_loader):04d}] "
                        f"[G: {last_loss_g.item():.4f}] "
                        f"[G_shape: {last_loss_g_shape.item():.4f}] "
                        f"[Reta: {last_loss_g_reta.item():.4f}] "
                        f"[Rphi: {last_loss_g_rphi.item():.4f}] "
                        f"[D_A: {loss_d_a.item():.4f}] "
                        f"[D_B: {loss_d_b.item():.4f}] "
                        f"[E: {last_loss_energy.item():.4f}]",
                        flush=True,
                    )
                else:
                    print(
                        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
                        f"[Batch {step:04d}/{len(train_loader):04d}] "
                        f"[G: {last_loss_g.item():.4f}] "
                        f"[D_A: {loss_d_a.item():.4f}] "
                        f"[D_B: {loss_d_b.item():.4f}] "
                        f"[E: {last_loss_energy.item():.4f}]",
                        flush=True,
                    )

        num_steps = max(len(train_loader), 1)
        num_gen_updates = max(gen_updates, 1)
        current_lr = optimizer_g.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch,
            "lr": current_lr,
            "loss_G": epoch_metrics["loss_G"] / num_gen_updates,
            "loss_G_GAN": epoch_metrics["loss_G_GAN"] / num_gen_updates,
            "loss_G_cycle": epoch_metrics["loss_G_cycle"] / num_gen_updates,
            "loss_G_identity": epoch_metrics["loss_G_identity"] / num_gen_updates,
            "loss_G_shape": epoch_metrics["loss_G_shape"] / num_gen_updates,
            "loss_G_reta": epoch_metrics["loss_G_reta"] / num_gen_updates,
            "loss_G_rphi": epoch_metrics["loss_G_rphi"] / num_gen_updates,
            "loss_G_energy": epoch_metrics["loss_G_energy"] / num_gen_updates,
            "loss_D_A": epoch_metrics["loss_D_A"] / num_steps,
            "loss_D_B": epoch_metrics["loss_D_B"] / num_steps,
            "generator_updates": gen_updates,
            "critic_updates": num_steps,
        }
        append_csv_row(log_path, log_row)

        scheduler_g.step()
        scheduler_d_a.step()
        scheduler_d_b.step()

        checkpoint_state = {
            "epoch": epoch,
            "config": config,
            "G_AB": generator_ab.state_dict(),
            "G_BA": generator_ba.state_dict(),
            "D_A": discriminator_a.state_dict(),
            "D_B": discriminator_b.state_dict(),
            "optimizer_G": optimizer_g.state_dict(),
            "optimizer_D_A": optimizer_d_a.state_dict(),
            "optimizer_D_B": optimizer_d_b.state_dict(),
            "scheduler_G": scheduler_g.state_dict(),
            "scheduler_D_A": scheduler_d_a.state_dict(),
            "scheduler_D_B": scheduler_d_b.state_dict(),
        }
        torch.save(checkpoint_state, checkpoint_dir / "latest.pth")
        if (args.save_interval > 0 and epoch % args.save_interval == 0) or epoch == total_epochs:
            torch.save(checkpoint_state, checkpoint_dir / f"epoch_{epoch:04d}.pth")

        if first_batch is not None and (
            args.sample_interval > 0 and (epoch % args.sample_interval == 0 or epoch == 1)
        ):
            if args.model_backend == "image":
                save_samples(generator_ab, generator_ba, first_batch, output_dir, epoch)
            run_sample_evaluation(
                generator_ab=generator_ab,
                model_backend=args.model_backend,
                electron_windows=electron_eval_windows,
                electron_conditions=electron_eval_conditions,
                electron_energy_scales=electron_eval_energy_scales,
                photon_windows=photon_eval_windows,
                photon_conditions=photon_eval_conditions,
                photon_energy_scales=photon_eval_energy_scales,
                sample_dir=output_dir / "samples" / f"epoch_{epoch:04d}",
                device=device,
                normalization_scale=float(args.normalization_scale),
                batch_size=int(args.sample_eval_batch_size),
                preprocessing=bundle.preprocessing,
                standardization=bundle.standardization,
                symmetric_range=symmetric_range,
                phi_window_size=int(args.phi_window_size),
                eta_window_size=int(args.eta_window_size),
            )
            print(
                f"Sample-stage evaluation written to {output_dir / 'samples' / f'epoch_{epoch:04d}'}",
                flush=True,
            )

    print(f"Training complete. Latest checkpoint: {checkpoint_dir / 'latest.pth'}", flush=True)


if __name__ == "__main__":
    main()
