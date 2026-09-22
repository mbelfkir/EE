from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from flow.utils import REPO_ROOT, ensure_ml_path


ensure_ml_path()

from data import (  # noqa: E402
    DEFAULT_ETA_WINDOW_SIZE,
    DEFAULT_IMAGE_TRANSFORM,
    DEFAULT_MIN_WINDOW_ENERGY_GEV,
    DEFAULT_PHI_WINDOW_SIZE,
    DEFAULT_USE_STANDARDIZATION,
)


DEFAULT_DATA_DIR = str((REPO_ROOT / "data").resolve())
DEFAULT_AUTOENCODER_OUTPUT_DIR = "outputs/latent_flow/autoencoder"
DEFAULT_FLOW_OUTPUT_DIR = "outputs/latent_flow/transport"


@dataclass
class AutoencoderDefaults:
    data_dir: str = DEFAULT_DATA_DIR
    output_dir: str = DEFAULT_AUTOENCODER_OUTPUT_DIR
    train_fraction: float = 0.8
    use_standardization: bool = DEFAULT_USE_STANDARDIZATION
    min_window_energy_gev: float = DEFAULT_MIN_WINDOW_ENERGY_GEV
    image_transform: str = DEFAULT_IMAGE_TRANSFORM
    eta_window_size: int = DEFAULT_ETA_WINDOW_SIZE
    phi_window_size: int = DEFAULT_PHI_WINDOW_SIZE
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 60
    epochs_decay: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    latent_dim: int = 16
    encoder_channels: tuple[int, ...] = (32, 64, 128)
    activation: str = "silu"
    norm: str = "group"
    output_activation: str = "softplus"
    l1_weight: float = 1.0
    l2_weight: float = 0.0
    latent_align_loss: str = "mmd"
    lambda_latent_align: float = 0.05
    selection_alpha: float = 0.1
    grad_clip: float = 5.0
    validation_interval: int = 1
    save_interval: int = 5
    sample_eval_count: int = 512
    sample_plot_count: int = 6
    resume: bool = False
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False


@dataclass
class FlowDefaults:
    data_dir: str = DEFAULT_DATA_DIR
    output_dir: str = DEFAULT_FLOW_OUTPUT_DIR
    train_fraction: float = 0.8
    use_standardization: bool = DEFAULT_USE_STANDARDIZATION
    min_window_energy_gev: float = DEFAULT_MIN_WINDOW_ENERGY_GEV
    image_transform: str = DEFAULT_IMAGE_TRANSFORM
    eta_window_size: int = DEFAULT_ETA_WINDOW_SIZE
    phi_window_size: int = DEFAULT_PHI_WINDOW_SIZE
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 80
    epochs_decay: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    latent_dim: int = 16
    encoder_channels: tuple[int, ...] = (32, 64, 128)
    activation: str = "silu"
    norm: str = "group"
    output_activation: str = "softplus"
    flow_hidden_dims: tuple[int, ...] = (128, 128)
    flow_blocks: int = 6
    domain_embedding_dim: int = 16
    scale_clamp: float = 1.5
    use_actnorm: bool = True
    lambda_mmd: float = 1.0
    lambda_moment: float = 0.5
    grad_clip: float = 5.0
    validation_interval: int = 5
    save_interval: int = 5
    sample_eval_count: int = 512
    sample_plot_count: int = 6
    resume: bool = False
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False


AUTOENCODER_DEFAULTS = AutoencoderDefaults()
FLOW_DEFAULTS = FlowDefaults()


def _positive_int_sequence(values: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(value) for value in values if int(value) > 0)
    if not dims:
        raise ValueError("Expected at least one positive integer.")
    return dims


def add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing the ROOT calorimeter files.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--use-standardization", dest="use_standardization", action="store_true")
    parser.add_argument("--no-standardization", dest="use_standardization", action="store_false")
    parser.set_defaults(use_standardization=DEFAULT_USE_STANDARDIZATION)
    parser.add_argument("--min-window-energy-gev", type=float, default=DEFAULT_MIN_WINDOW_ENERGY_GEV)
    parser.add_argument("--image-transform", choices=["none", "cbrt"], default=DEFAULT_IMAGE_TRANSFORM)
    parser.add_argument("--eta-window-size", type=int, default=DEFAULT_ETA_WINDOW_SIZE)
    parser.add_argument("--phi-window-size", type=int, default=DEFAULT_PHI_WINDOW_SIZE)


def add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epochs-decay", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--sample-eval-count", type=int, default=512)
    parser.add_argument("--sample-plot-count", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch algorithms where possible.")


def build_autoencoder_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: train a convolutional autoencoder on the same calorimeter inputs "
            "used by the shared ML loader, preserving the existing Cal_e image scaling."
        )
    )
    add_shared_dataset_args(parser)
    parser.add_argument("--output-dir", default=DEFAULT_AUTOENCODER_OUTPUT_DIR, help="Directory for checkpoints, logs, and validation plots.")
    parser.add_argument("--batch-size", type=int, default=AUTOENCODER_DEFAULTS.batch_size)
    parser.add_argument("--num-workers", type=int, default=AUTOENCODER_DEFAULTS.num_workers)
    parser.add_argument("--epochs", type=int, default=AUTOENCODER_DEFAULTS.epochs)
    parser.add_argument("--epochs-decay", type=int, default=AUTOENCODER_DEFAULTS.epochs_decay)
    parser.add_argument("--lr", type=float, default=AUTOENCODER_DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=AUTOENCODER_DEFAULTS.weight_decay)
    parser.add_argument("--beta1", type=float, default=AUTOENCODER_DEFAULTS.beta1)
    parser.add_argument("--beta2", type=float, default=AUTOENCODER_DEFAULTS.beta2)
    parser.add_argument("--latent-dim", type=int, default=AUTOENCODER_DEFAULTS.latent_dim)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=list(AUTOENCODER_DEFAULTS.encoder_channels))
    parser.add_argument("--activation", choices=["relu", "lrelu", "gelu", "silu"], default=AUTOENCODER_DEFAULTS.activation)
    parser.add_argument("--norm", choices=["none", "batch", "instance", "group"], default=AUTOENCODER_DEFAULTS.norm)
    parser.add_argument("--output-activation", choices=["identity", "relu", "softplus"], default=AUTOENCODER_DEFAULTS.output_activation)
    parser.add_argument("--l1-weight", type=float, default=AUTOENCODER_DEFAULTS.l1_weight)
    parser.add_argument("--l2-weight", type=float, default=AUTOENCODER_DEFAULTS.l2_weight)
    parser.add_argument("--latent-align-loss", choices=["mmd", "coral"], default=AUTOENCODER_DEFAULTS.latent_align_loss)
    parser.add_argument("--lambda-latent-align", type=float, default=AUTOENCODER_DEFAULTS.lambda_latent_align)
    parser.add_argument("--selection-alpha", type=float, default=AUTOENCODER_DEFAULTS.selection_alpha)
    parser.add_argument("--grad-clip", type=float, default=AUTOENCODER_DEFAULTS.grad_clip)
    parser.add_argument("--validation-interval", type=int, default=AUTOENCODER_DEFAULTS.validation_interval)
    parser.add_argument("--save-interval", type=int, default=AUTOENCODER_DEFAULTS.save_interval)
    parser.add_argument("--sample-eval-count", type=int, default=AUTOENCODER_DEFAULTS.sample_eval_count)
    parser.add_argument("--sample-plot-count", type=int, default=AUTOENCODER_DEFAULTS.sample_plot_count)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=AUTOENCODER_DEFAULTS.device)
    parser.add_argument("--seed", type=int, default=AUTOENCODER_DEFAULTS.seed)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def build_flow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: train a latent-domain transport flow on frozen autoencoder latents "
            "for unpaired electron-to-photon calorimeter translation."
        )
    )
    add_shared_dataset_args(parser)
    parser.add_argument("--autoencoder-checkpoint", required=True, help="Stage-1 checkpoint containing the trained encoder and decoder.")
    parser.add_argument("--output-dir", default=DEFAULT_FLOW_OUTPUT_DIR, help="Directory for flow checkpoints, logs, and validation plots.")
    parser.add_argument("--batch-size", type=int, default=FLOW_DEFAULTS.batch_size)
    parser.add_argument("--num-workers", type=int, default=FLOW_DEFAULTS.num_workers)
    parser.add_argument("--epochs", type=int, default=FLOW_DEFAULTS.epochs)
    parser.add_argument("--epochs-decay", type=int, default=FLOW_DEFAULTS.epochs_decay)
    parser.add_argument("--lr", type=float, default=FLOW_DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=FLOW_DEFAULTS.weight_decay)
    parser.add_argument("--beta1", type=float, default=FLOW_DEFAULTS.beta1)
    parser.add_argument("--beta2", type=float, default=FLOW_DEFAULTS.beta2)
    parser.add_argument("--latent-dim", type=int, default=FLOW_DEFAULTS.latent_dim)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=list(FLOW_DEFAULTS.encoder_channels))
    parser.add_argument("--activation", choices=["relu", "lrelu", "gelu", "silu"], default=FLOW_DEFAULTS.activation)
    parser.add_argument("--norm", choices=["none", "batch", "instance", "group"], default=FLOW_DEFAULTS.norm)
    parser.add_argument("--output-activation", choices=["identity", "relu", "softplus"], default=FLOW_DEFAULTS.output_activation)
    parser.add_argument("--flow-hidden-dims", type=int, nargs="+", default=list(FLOW_DEFAULTS.flow_hidden_dims))
    parser.add_argument("--flow-blocks", type=int, default=FLOW_DEFAULTS.flow_blocks)
    parser.add_argument("--domain-embedding-dim", type=int, default=FLOW_DEFAULTS.domain_embedding_dim)
    parser.add_argument("--scale-clamp", type=float, default=FLOW_DEFAULTS.scale_clamp)
    parser.add_argument("--no-actnorm", dest="use_actnorm", action="store_false")
    parser.add_argument("--use-actnorm", dest="use_actnorm", action="store_true")
    parser.set_defaults(use_actnorm=FLOW_DEFAULTS.use_actnorm)
    parser.add_argument("--lambda-mmd", type=float, default=FLOW_DEFAULTS.lambda_mmd)
    parser.add_argument("--lambda-moment", type=float, default=FLOW_DEFAULTS.lambda_moment)
    parser.add_argument("--grad-clip", type=float, default=FLOW_DEFAULTS.grad_clip)
    parser.add_argument("--validation-interval", type=int, default=FLOW_DEFAULTS.validation_interval)
    parser.add_argument("--save-interval", type=int, default=FLOW_DEFAULTS.save_interval)
    parser.add_argument("--sample-eval-count", type=int, default=FLOW_DEFAULTS.sample_eval_count)
    parser.add_argument("--sample-plot-count", type=int, default=FLOW_DEFAULTS.sample_plot_count)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=FLOW_DEFAULTS.device)
    parser.add_argument("--seed", type=int, default=FLOW_DEFAULTS.seed)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def canonicalise_autoencoder_args(args: argparse.Namespace) -> argparse.Namespace:
    args.encoder_channels = list(_positive_int_sequence(args.encoder_channels))
    args.latent_dim = int(args.latent_dim)
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive.")
    args.latent_align_loss = str(args.latent_align_loss).lower()
    if args.latent_align_loss not in {"mmd", "coral"}:
        raise ValueError("--latent-align-loss must be one of: mmd, coral.")
    if float(args.lambda_latent_align) < 0.0:
        raise ValueError("--lambda-latent-align must be non-negative.")
    if float(args.selection_alpha) < 0.0:
        raise ValueError("--selection-alpha must be non-negative.")
    return args


def canonicalise_flow_args(args: argparse.Namespace) -> argparse.Namespace:
    args.encoder_channels = list(_positive_int_sequence(args.encoder_channels))
    args.flow_hidden_dims = list(_positive_int_sequence(args.flow_hidden_dims))
    args.latent_dim = int(args.latent_dim)
    if args.latent_dim <= 1:
        raise ValueError("--latent-dim must be at least 2 for the affine coupling flow.")
    return args
