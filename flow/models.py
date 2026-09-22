from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LOG_2PI = math.log(2.0 * math.pi)


def valid_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def build_norm(name: str, num_channels: int) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "none":
        return nn.Identity()
    if normalized == "batch":
        return nn.BatchNorm2d(num_channels)
    if normalized == "instance":
        return nn.InstanceNorm2d(num_channels)
    if normalized == "group":
        return nn.GroupNorm(valid_group_count(num_channels), num_channels)
    raise ValueError(f"Unsupported normalization '{name}'.")


def build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}'.")


def build_output_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "identity":
        return nn.Identity()
    if normalized == "relu":
        return nn.ReLU(inplace=False)
    if normalized == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported output activation '{name}'.")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        norm: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            build_norm(norm, out_channels),
            build_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=2, norm=norm, activation=activation),
            ConvNormAct(out_channels, out_channels, stride=1, norm=norm, activation=activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=1, norm=norm, activation=activation),
            ConvNormAct(out_channels, out_channels, stride=1, norm=norm, activation=activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvolutionalEncoder(nn.Module):
    """CNN encoder for calorimeter windows with a configurable latent dimension."""

    def __init__(
        self,
        input_shape: Sequence[int],
        latent_dim: int = 16,
        channels: Sequence[int] = (32, 64, 128),
        norm: str = "group",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if len(tuple(input_shape)) != 3:
            raise ValueError(f"input_shape must be [channels, phi, eta], got {tuple(input_shape)}")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        channel_schedule = tuple(int(value) for value in channels if int(value) > 0)
        if not channel_schedule:
            raise ValueError("channels must contain at least one positive integer.")

        self.input_shape = tuple(int(value) for value in input_shape)
        self.latent_dim = int(latent_dim)
        self.channels = channel_schedule
        input_channels = self.input_shape[0]

        self.stem = ConvNormAct(input_channels, self.channels[0], stride=1, norm=norm, activation=activation)
        self.blocks = nn.ModuleList(
            [
                ConvEncoderBlock(self.channels[index], self.channels[index + 1], norm=norm, activation=activation)
                for index in range(len(self.channels) - 1)
            ]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            x = self.stem(dummy)
            feature_shapes = [tuple(int(value) for value in x.shape[1:])]
            for block in self.blocks:
                x = block(x)
                feature_shapes.append(tuple(int(value) for value in x.shape[1:]))
            self.feature_shape = feature_shapes[-1]
            self.feature_shapes = tuple(feature_shapes)
            self.flattened_dim = int(x.reshape(1, -1).shape[1])

        self.to_latent = nn.Linear(self.flattened_dim, self.latent_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        y = self.stem(x)
        for block in self.blocks:
            y = block(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.to_latent(features.flatten(start_dim=1))


class ConvolutionalDecoder(nn.Module):
    """Mirror decoder that reconstructs a calorimeter image from a latent vector."""

    def __init__(
        self,
        output_shape: Sequence[int],
        latent_dim: int,
        channels: Sequence[int],
        encoder_feature_shapes: Sequence[Sequence[int]],
        norm: str = "group",
        activation: str = "silu",
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        output_shape_tuple = tuple(int(value) for value in output_shape)
        if len(output_shape_tuple) != 3:
            raise ValueError(f"output_shape must be [channels, phi, eta], got {output_shape_tuple}")
        channel_schedule = tuple(int(value) for value in channels if int(value) > 0)
        if not channel_schedule:
            raise ValueError("channels must contain at least one positive integer.")
        if len(encoder_feature_shapes) != len(channel_schedule):
            raise ValueError(
                "encoder_feature_shapes must match the encoder channel schedule length: "
                f"{len(encoder_feature_shapes)} vs {len(channel_schedule)}"
            )

        self.output_shape = output_shape_tuple
        self.feature_shape = tuple(int(value) for value in encoder_feature_shapes[-1])
        self.feature_shapes = [tuple(int(value) for value in shape) for shape in encoder_feature_shapes]
        self.from_latent = nn.Linear(int(latent_dim), int(math.prod(self.feature_shape)))
        self.target_shapes = list(reversed(self.feature_shapes[:-1]))

        reversed_channels = list(reversed(channel_schedule[:-1]))
        current_channels = self.feature_shape[0]
        self.up_blocks = nn.ModuleList()
        for out_channels in reversed_channels:
            self.up_blocks.append(ConvDecoderBlock(current_channels, out_channels, norm=norm, activation=activation))
            current_channels = out_channels

        mid_channels = max(current_channels, self.output_shape[0] * 2)
        self.output_block = nn.Sequential(
            ConvNormAct(current_channels, mid_channels, stride=1, norm=norm, activation=activation),
            nn.Conv2d(mid_channels, self.output_shape[0], kernel_size=3, stride=1, padding=1),
            build_output_activation(output_activation),
        )

    def forward(self, latent: Tensor) -> Tensor:
        x = self.from_latent(latent).view(latent.size(0), *self.feature_shape)
        for block, target_shape in zip(self.up_blocks, self.target_shapes):
            x = F.interpolate(x, size=target_shape[1:], mode="bilinear", align_corners=False)
            x = block(x)
        x = F.interpolate(x, size=self.output_shape[1:], mode="bilinear", align_corners=False)
        return self.output_block(x)


class LatentAutoencoder(nn.Module):
    """Autoencoder wrapper exposing separate encoder and decoder modules."""

    def __init__(self, encoder: ConvolutionalEncoder, decoder: ConvolutionalDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def reconstruct(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def build_autoencoder(
    image_shape: Sequence[int],
    latent_dim: int = 16,
    channels: Sequence[int] = (32, 64, 128),
    norm: str = "group",
    activation: str = "silu",
    output_activation: str = "softplus",
) -> LatentAutoencoder:
    encoder = ConvolutionalEncoder(
        input_shape=image_shape,
        latent_dim=latent_dim,
        channels=channels,
        norm=norm,
        activation=activation,
    )
    decoder = ConvolutionalDecoder(
        output_shape=image_shape,
        latent_dim=latent_dim,
        channels=channels,
        encoder_feature_shapes=encoder.feature_shapes,
        norm=norm,
        activation=activation,
        output_activation=output_activation,
    )
    return LatentAutoencoder(encoder=encoder, decoder=decoder)


def build_autoencoder_from_config(config: Mapping[str, object]) -> LatentAutoencoder:
    phi_window_size = int(config["phi_window_size"])
    eta_window_size = int(config["eta_window_size"])
    latent_dim = int(config["latent_dim"])
    channels = tuple(int(value) for value in config["encoder_channels"])
    activation = str(config.get("activation", "silu"))
    norm = str(config.get("norm", "group"))
    output_activation = str(config.get("output_activation", "softplus"))
    return build_autoencoder(
        image_shape=(1, phi_window_size, eta_window_size),
        latent_dim=latent_dim,
        channels=channels,
        norm=norm,
        activation=activation,
        output_activation=output_activation,
    )


class ActNorm1d(nn.Module):
    """Per-dimension affine normalisation with data-dependent initialisation."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def initialize(self, x: Tensor) -> None:
        if self.initialized.item():
            return
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            self.bias.copy_(-mean)
            self.log_scale.copy_(torch.log(1.0 / std))
            self.initialized.fill_(True)

    def forward(self, x: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError(f"ActNorm1d expects [batch, dim] tensors, got {tuple(x.shape)}")
        if not self.initialized.item():
            self.initialize(x)
        log_det = self.log_scale.sum(dim=1 if self.log_scale.ndim > 1 else 0)
        if reverse:
            y = x * torch.exp(-self.log_scale) - self.bias
            return y, -log_det.expand(x.size(0))
        y = (x + self.bias) * torch.exp(self.log_scale)
        return y, log_det.expand(x.size(0))


class InvertiblePermutation(nn.Module):
    def __init__(self, num_features: int, reverse_order: bool = False) -> None:
        super().__init__()
        if reverse_order:
            permutation = torch.arange(num_features - 1, -1, -1)
        else:
            permutation = torch.roll(torch.arange(num_features), shifts=1)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(num_features)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        if reverse:
            return x[:, self.inverse_permutation]
        return x[:, self.permutation]


class CouplingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        hidden = [int(value) for value in hidden_dims if int(value) > 0]
        if not hidden:
            raise ValueError("hidden_dims must contain at least one positive integer.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
            current_dim = hidden_dim
        final = nn.Linear(current_dim, output_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class AffineCoupling1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int],
        condition_dim: int,
        mask: Tensor,
        scale_clamp: float = 1.5,
    ) -> None:
        super().__init__()
        if mask.ndim != 1 or mask.numel() != num_features:
            raise ValueError("mask must be a 1D tensor with one entry per latent feature.")
        self.num_features = int(num_features)
        self.condition_dim = int(condition_dim)
        self.scale_clamp = float(scale_clamp)
        self.register_buffer("mask", mask.view(1, num_features).float())
        self.network = CouplingMLP(
            input_dim=self.num_features + self.condition_dim,
            hidden_dims=hidden_dims,
            output_dim=self.num_features * 2,
        )

    def forward(self, x: Tensor, condition: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        masked = x * self.mask
        network_input = torch.cat([masked, condition], dim=1)
        log_scale, shift = self.network(network_input).chunk(2, dim=1)
        inactive = 1.0 - self.mask
        log_scale = self.scale_clamp * torch.tanh(log_scale / self.scale_clamp) * inactive
        shift = shift * inactive
        if reverse:
            y = masked + inactive * ((x - shift) * torch.exp(-log_scale))
            return y, -log_scale.sum(dim=1)
        y = masked + inactive * (x * torch.exp(log_scale) + shift)
        return y, log_scale.sum(dim=1)


class ConditionalLatentFlow(nn.Module):
    """Conditional RealNVP-style latent transport model operating on vectors only."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        num_blocks: int = 6,
        num_domains: int = 2,
        domain_embedding_dim: int = 16,
        scale_clamp: float = 1.5,
        use_actnorm: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 1:
            raise ValueError("latent_dim must be at least 2 for affine coupling.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        self.latent_dim = int(latent_dim)
        self.num_domains = int(num_domains)
        self.domain_embedding = nn.Embedding(self.num_domains, int(domain_embedding_dim))
        self.permutations = nn.ModuleList()
        self.actnorm_layers = nn.ModuleList()
        self.coupling_layers = nn.ModuleList()

        for block_index in range(int(num_blocks)):
            base_mask = torch.tensor(
                [(feature_index + block_index) % 2 for feature_index in range(self.latent_dim)],
                dtype=torch.float32,
            )
            self.permutations.append(InvertiblePermutation(self.latent_dim, reverse_order=bool(block_index % 2)))
            if use_actnorm:
                self.actnorm_layers.append(ActNorm1d(self.latent_dim))
            else:
                self.actnorm_layers.append(nn.Identity())  # type: ignore[arg-type]
            self.coupling_layers.append(
                AffineCoupling1d(
                    num_features=self.latent_dim,
                    hidden_dims=hidden_dims,
                    condition_dim=int(domain_embedding_dim),
                    mask=base_mask,
                    scale_clamp=scale_clamp,
                )
            )

    def _domain_tensor(self, domain: int | Tensor, batch_size: int, device: torch.device) -> Tensor:
        if isinstance(domain, Tensor):
            if domain.ndim == 0:
                domain = domain.repeat(batch_size)
            if domain.ndim != 1:
                raise ValueError(f"Domain tensor must be scalar or [batch], got {tuple(domain.shape)}")
            if domain.size(0) != batch_size:
                raise ValueError(f"Domain tensor batch mismatch: {domain.size(0)} vs {batch_size}")
            return domain.to(device=device, dtype=torch.long)
        return torch.full((batch_size,), int(domain), device=device, dtype=torch.long)

    def _condition(self, domain: int | Tensor, batch_size: int, device: torch.device) -> Tensor:
        domain_tensor = self._domain_tensor(domain, batch_size=batch_size, device=device)
        return self.domain_embedding(domain_tensor)

    def forward(self, x: Tensor, domain: int | Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError(f"ConditionalLatentFlow expects [batch, dim] tensors, got {tuple(x.shape)}")
        condition = self._condition(domain, batch_size=x.size(0), device=x.device)
        z = x
        log_det_total = x.new_zeros(x.size(0))
        for permutation, actnorm, coupling in zip(self.permutations, self.actnorm_layers, self.coupling_layers):
            z = permutation(z, reverse=False)
            if isinstance(actnorm, ActNorm1d):
                z, log_det = actnorm(z, reverse=False)
                log_det_total = log_det_total + log_det
            z, log_det = coupling(z, condition, reverse=False)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z: Tensor, domain: int | Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"ConditionalLatentFlow expects [batch, dim] tensors, got {tuple(z.shape)}")
        condition = self._condition(domain, batch_size=z.size(0), device=z.device)
        x = z
        for permutation, actnorm, coupling in zip(
            reversed(self.permutations),
            reversed(self.actnorm_layers),
            reversed(self.coupling_layers),
        ):
            x, _ = coupling(x, condition, reverse=True)
            if isinstance(actnorm, ActNorm1d):
                x, _ = actnorm(x, reverse=True)
            x = permutation(x, reverse=True)
        return x

    def log_prob(self, x: Tensor, domain: int | Tensor) -> Tensor:
        z, log_det = self.forward(x, domain)
        log_base = -0.5 * (z.pow(2) + LOG_2PI).sum(dim=1)
        return log_base + log_det

    def transport(self, x: Tensor, source_domain: int | Tensor, target_domain: int | Tensor) -> Tensor:
        latent, _ = self.forward(x, source_domain)
        return self.inverse(latent, target_domain)


class LatentTransportModel(nn.Module):
    """Convenience wrapper joining the encoder, decoder, and latent flow."""

    def __init__(self, autoencoder: LatentAutoencoder, flow: ConditionalLatentFlow) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.flow = flow

    def encode(self, x: Tensor) -> Tensor:
        return self.autoencoder.encode(x)

    def decode(self, latent: Tensor) -> Tensor:
        return self.autoencoder.decode(latent)

    def reconstruct(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.autoencoder.reconstruct(x)

    def translate(self, x: Tensor, source_domain: int | Tensor, target_domain: int | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.encode(x)
        transported = self.flow.transport(latent, source_domain=source_domain, target_domain=target_domain)
        translated = self.decode(transported)
        return translated, latent, transported
