from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils import spectral_norm
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/models.py requires torch. Install the packages listed in ML/requirements.txt."
    ) from exc


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        ngf: int = 64,
        n_blocks: int = 6,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive.")

        self.input_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        in_features = ngf
        out_features = in_features * 2
        self.down_blocks = nn.ModuleList()
        for _ in range(3):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                )
            )
            in_features = out_features
            out_features = in_features * 2

        self.res_blocks = nn.Sequential(*[ResidualBlock(in_features) for _ in range(n_blocks)])

        out_features = in_features // 2
        self.up_convs = nn.ModuleList()
        self.up_norms = nn.ModuleList()
        self.up_activations = nn.ModuleList()
        for _ in range(3):
            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )
            )
            self.up_norms.append(nn.InstanceNorm2d(out_features))
            self.up_activations.append(nn.ReLU(inplace=True))
            in_features = out_features
            out_features = in_features // 2

        self.output_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        output_sizes = []

        x = self.input_block(x)
        output_sizes.append(x.shape)

        for down_block in self.down_blocks:
            x = down_block(x)
            output_sizes.append(x.shape)

        x = self.res_blocks(x)

        target_shapes = list(reversed(output_sizes[:-1]))
        for up_conv, up_norm, up_activation, target_shape in zip(
            self.up_convs,
            self.up_norms,
            self.up_activations,
            target_shapes,
        ):
            x = up_conv(x, output_size=target_shape)
            x = up_norm(x)
            x = up_activation(x)

        return self.output_block(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        # Use 3x3 kernels so small calorimeter windows like 7x11 and 11x11
        # keep valid spatial support through the PatchGAN stack.
        kernel_size = 3
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult_prev = 1
        nf_mult = 1
        for layer_index in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_index, 8)
            sequence.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding),
            ]
        )

        self.model = nn.Sequential(*sequence)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class GlobalDiscriminator(nn.Module):
    """
    Global discriminator-style classifier for calorimeter windows.

    This reuses the same PatchGAN convolutional trunk as the CycleGAN image
    discriminator, then averages the patch-score map down to a single scalar
    score per image so the network can be trained as an electron/photon
    classifier with the same MSE target convention.
    """

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.patch_discriminator = PatchGANDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        patch_scores = self.patch_discriminator(x)
        return self.pool(patch_scores).view(x.size(0), -1)


def parse_hidden_dims(values: tuple[int, ...] | list[int] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        values = default
    dims = tuple(int(value) for value in values if int(value) > 0)
    if not dims:
        raise ValueError("Hidden-dimension list must contain at least one positive integer.")
    return dims


def build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized in {"relu"}:
        return nn.ReLU(inplace=True)
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized in {"lrelu", "leaky_relu"}:
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def replace_voxel_energy(voxel: Tensor, energy: Tensor) -> Tensor:
    if voxel.ndim != 3:
        raise ValueError(f"Expected voxel tensor with shape [batch, num_voxels, num_features], got {tuple(voxel.shape)}")
    updated = voxel.clone()
    updated[:, :, 0] = energy
    return updated


class VoxelGenerator(nn.Module):
    """
    Paper-inspired dense generator.

    Architecture follows the generator pattern described in arXiv:2309.06515:
    three hidden dense layers with batch normalization and a smooth activation.
    For translation, the input is the flattened voxel representation plus a
    two-component shower-shape condition, and the output is the target-domain
    voxel energy vector.
    """

    def __init__(
        self,
        num_voxels: int,
        feature_dim: int = 5,
        condition_dim: int = 2,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.num_voxels = int(num_voxels)
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dims = parse_hidden_dims(hidden_dims, default=(100, 200, 400))
        self.activation_name = str(activation)

        layers: list[nn.Module] = []
        input_dim = self.num_voxels * self.feature_dim + self.condition_dim
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    build_activation(self.activation_name),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.num_voxels))
        self.model = nn.Sequential(*layers)
        self.output_activation = nn.Softplus()

    def forward(self, voxel: Tensor, condition: Tensor) -> Tensor:
        if voxel.ndim != 3:
            raise ValueError(f"VoxelGenerator expects [batch, num_voxels, num_features], got {tuple(voxel.shape)}")
        flattened = voxel.reshape(voxel.size(0), -1)
        conditioned = torch.cat([flattened, condition.reshape(condition.size(0), -1)], dim=1)
        return self.output_activation(self.model(conditioned))


class VoxelCritic(nn.Module):
    """
    Paper-inspired dense critic.

    Architecture follows the critic/discriminator pattern described in
    arXiv:2309.06515: three dense layers with ReLU activations and spectral
    normalization, conditioned on a scalar energy label.
    """

    def __init__(
        self,
        num_voxels: int,
        feature_dim: int = 5,
        condition_dim: int = 2,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_voxels = int(num_voxels)
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dims = parse_hidden_dims(hidden_dims, default=(400, 200, 100))

        layers: list[nn.Module] = []
        input_dim = self.num_voxels * self.feature_dim + self.condition_dim
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    spectral_norm(nn.Linear(previous_dim, hidden_dim)),
                    nn.ReLU(inplace=True),
                ]
            )
            previous_dim = hidden_dim
        layers.append(spectral_norm(nn.Linear(previous_dim, 1)))
        self.model = nn.Sequential(*layers)

    def forward(self, voxel: Tensor, condition: Tensor) -> Tensor:
        if voxel.ndim != 3:
            raise ValueError(f"VoxelCritic expects [batch, num_voxels, num_features], got {tuple(voxel.shape)}")
        flattened = voxel.reshape(voxel.size(0), -1)
        conditioned = torch.cat([flattened, condition.reshape(condition.size(0), -1)], dim=1)
        return self.model(conditioned)
