from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parent.parent
FLOW_DIR = REPO_ROOT / "flow"
ML_DIR = REPO_ROOT / "ML"
ELECTRON_DOMAIN = "electron"
PHOTON_DOMAIN = "photon"
DOMAIN_TO_INDEX = {ELECTRON_DOMAIN: 0, PHOTON_DOMAIN: 1}
INDEX_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_INDEX.items()}


def ensure_ml_path() -> Path:
    """Expose the legacy ML modules on ``sys.path`` for reuse."""

    ml_dir_str = str(ML_DIR)
    if ml_dir_str not in sys.path:
        sys.path.insert(0, ml_dir_str)
    return ML_DIR


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed data-loader workers consistently."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def choose_device(device_name: str = "auto") -> torch.device:
    """Pick a sensible training device."""

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def serialise_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert dataclasses and argparse namespaces into JSON-safe dictionaries."""

    if is_dataclass(config):
        payload = asdict(config)
    elif hasattr(config, "__dict__"):
        payload = dict(vars(config))
    else:
        payload = dict(config)

    serialised: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serialised[key] = str(value)
        elif isinstance(value, tuple):
            serialised[key] = list(value)
        elif isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        else:
            serialised[key] = value
    return serialised


def save_json(data: Mapping[str, Any] | Sequence[Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))


def latest_checkpoint_path(checkpoint_dir: str | Path, extension: str = ".pt") -> Path | None:
    directory = Path(checkpoint_dir)
    latest = directory / f"latest{extension}"
    if latest.exists():
        return latest
    checkpoints = sorted(directory.glob(f"epoch_*{extension}"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def build_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    n_epochs_decay: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if n_epochs <= 0 or n_epochs_decay < 0:
        raise ValueError("n_epochs must be positive and n_epochs_decay must be non-negative.")

    if n_epochs_decay == 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lambda_rule(epoch: int) -> float:
        return 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def set_requires_grad(modules: nn.Module | Sequence[nn.Module], requires_grad: bool) -> None:
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def tensor_batch_to_numpy(batch: Tensor) -> np.ndarray:
    array = batch.detach().cpu().numpy().astype(np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    return array


def to_physical_images(images: np.ndarray, energy_scales: np.ndarray) -> np.ndarray:
    """Undo the simple-pipeline image scaling by multiplying with ``Cal_e``."""

    image_array = np.asarray(images, dtype=np.float32)
    scale_array = np.asarray(energy_scales, dtype=np.float32).reshape(-1, 1, 1)
    if image_array.ndim != 3:
        raise ValueError(f"Expected image array with shape [N, phi, eta], got {image_array.shape}")
    if scale_array.shape[0] != image_array.shape[0]:
        raise ValueError(
            "Energy-scale length does not match the number of images: "
            f"{scale_array.shape[0]} vs {image_array.shape[0]}"
        )
    return (image_array * scale_array).astype(np.float32)


def domain_to_index(domain: str) -> int:
    if domain not in DOMAIN_TO_INDEX:
        raise ValueError(f"Unsupported domain '{domain}'. Expected one of {sorted(DOMAIN_TO_INDEX)}.")
    return DOMAIN_TO_INDEX[domain]


def index_to_domain(index: int) -> str:
    if int(index) not in INDEX_TO_DOMAIN:
        raise ValueError(f"Unsupported domain index '{index}'.")
    return INDEX_TO_DOMAIN[int(index)]


def infinite_loader(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def specified_cli_flags(argv: Sequence[str] | None = None) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    flags: set[str] = set()
    for token in tokens:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags
