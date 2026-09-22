from __future__ import annotations

import csv
import json
import random
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/utils.py requires torch and numpy. Install the packages listed in ML/requirements.txt."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("ML/cyclegan_ep/utils.py requires Pillow for PNG export.") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.InstanceNorm2d) and module.weight is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        if max_size <= 0:
            raise ValueError("ReplayBuffer max_size must be positive.")
        self.max_size = max_size
        self.data: deque[Tensor] = deque()

    def push_and_pop(self, batch: Tensor) -> Tensor:
        items = []
        for element in batch.detach():
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                items.append(element)
            elif random.random() > 0.5:
                idx = random.randrange(len(self.data))
                stored = self.data[idx].clone()
                self.data[idx] = element
                items.append(stored)
            else:
                items.append(element)
        return torch.cat(items, dim=0)


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


def set_requires_grad(modules: nn.Module | Sequence[nn.Module], requires_grad: bool) -> None:
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def normalized_to_unit_interval(tensor: Tensor) -> Tensor:
    return (tensor + 1.0) * 0.5


def normalized_to_raw_tensor(
    tensor: Tensor,
    normalization_scale: float,
    symmetric_range: bool = False,
) -> Tensor:
    if symmetric_range:
        return tensor.clamp(-1.0, 1.0) * normalization_scale
    return normalized_to_unit_interval(tensor) * normalization_scale


def normalized_to_raw_numpy(
    tensor: Tensor,
    normalization_scale: float,
    symmetric_range: bool = False,
) -> np.ndarray:
    if symmetric_range:
        array = tensor.detach().cpu().clamp(-1.0, 1.0).squeeze().numpy()
        return (array * normalization_scale).astype(np.float32)
    array = normalized_to_unit_interval(tensor.detach().cpu()).clamp(0.0, 1.0).squeeze().numpy()
    return (array * normalization_scale).astype(np.float32)


def raw_to_normalized_numpy(
    array: np.ndarray,
    normalization_scale: float,
    symmetric_range: bool = False,
) -> np.ndarray:
    if symmetric_range:
        clipped = np.clip(array.astype(np.float32), -normalization_scale, normalization_scale)
        return clipped / normalization_scale
    clipped = np.clip(array.astype(np.float32), 0.0, normalization_scale)
    return 2.0 * (clipped / normalization_scale) - 1.0


def save_tensor_png(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    image = normalized_to_unit_interval(tensor.detach().cpu()).clamp(0.0, 1.0).squeeze().numpy()
    image_uint8 = np.rint(image * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def save_window_png(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"save_window_png expects a 2D array, got shape {image.shape}")
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        normalized = np.zeros_like(image, dtype=np.float32)
    else:
        normalized = (image - minimum) / (maximum - minimum)
    image_uint8 = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def append_csv_row(path: str | Path, row: dict[str, float | int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def latest_checkpoint_path(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pth"
    if latest.exists():
        return latest
    epoch_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"))
    if not epoch_checkpoints:
        return None
    return epoch_checkpoints[-1]
