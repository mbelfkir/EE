# Latent image flow

This package implements two-stage unpaired electron-to-photon image transport.

1. `train_autoencoder.py` trains a convolutional autoencoder on layer-2
   calorimeter windows.
2. `train_flow.py` freezes the autoencoder and learns conditional latent
   transport with a RealNVP-style flow.

The package reuses `ML/data.py` and keeps raw image axes as `[phi, eta]` and
model tensors as `[batch, channel, phi, eta]`.

Train stage 1:

```bash
python -m flow.train_autoencoder \
  --data-dir data \
  --output-dir outputs/latent_flow/autoencoder \
  --phi-window-size 11 \
  --eta-window-size 11 \
  --latent-dim 32
```

Train stage 2:

```bash
python -m flow.train_flow \
  --autoencoder-checkpoint outputs/latent_flow/autoencoder/checkpoints/best.pt \
  --output-dir outputs/latent_flow/transport \
  --flow-blocks 6 \
  --flow-hidden-dims 128 128
```

Validation outputs include reconstruction and translation examples, mean
windows, total-energy and shower-shape histograms, latent comparisons, and JSON
metric summaries. Checkpoint selection should prioritize physical validation
metrics over training loss alone.

Key modules:

- `models.py`: autoencoder and latent-flow models.
- `datasets.py`: PyTorch datasets over the shared ROOT loader.
- `losses.py`: reconstruction and latent-alignment losses.
- `validate.py`: physics metrics and validation exports.
- `plotting.py`: matplotlib plotting helpers.
- `config.py`: command-line defaults.
