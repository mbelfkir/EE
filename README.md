# Electron-to-Photon Calorimeter Translation

This repository contains the dataset, maintained training pipelines, physics
plots, and dataset-inspection utilities for electron-to-photon calorimeter
domain translation.

The training task is unpaired: electron and photon events are sampled from
separate domains. Model quality should therefore be judged from total-energy,
mean-window, core-window, and shower-shape distributions rather than from
event-by-event electron/photon agreement.

## Repository layout

```text
EE/
├── data/                       ROOT calorimeter ntuples and dataset notes
├── ML/                         Shared window loader and CycleGAN pipeline
│   └── cyclegan_ep/            CycleGAN training, inference, and evaluation
├── flow/                       Autoencoder plus latent normalizing flow
├── flow_electron_to_photon/    Shower-shape conditional normalizing flow
├── scripts/                    ROOT and Python physics plotting programs
├── simulation/                 Geant4 source, production macros, and audit tool
└── outputs/                    Generated runs and plots, ignored by Git
```

The clean copy intentionally excludes old UNIT experiments, HybridFLGAN,
duplicate prototype pipelines, smoke-test runs, checkpoints, generated plots,
virtual environments, caches, and presentation files.

## Dataset

The ROOT files use a `physics` tree. Important branches include cell energy,
position, dimensions, and layer; ECAL and HCAL layer-energy sums; calorimeter
energy summaries; and particle truth.

The four files used by the main configs each contain 10,000 events. A direct
audit shows that all four contain particle energies from approximately 10 to
50 GeV with means near 30 GeV. The `30GeV` and `50GeV` filename tokens should
therefore be treated as nominal sample labels, not fixed truth energies. See
[data/README.md](data/README.md) and run the audit before training:

```bash
python simulation/inspect_samples.py data
```

ROOT files are configured for Git LFS in `.gitattributes`.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The C++ plotting macros additionally require CERN ROOT with the `root`
executable available on `PATH`. Building the detector simulation requires
Geant4 with ROOT analysis support; see
[simulation/geant4/README.md](simulation/geant4/README.md).

## Training

### CycleGAN

The CycleGAN supports image and voxel backends. Calorimeter windows use
`[phi, eta]`; model tensors use `[batch, channel, phi, eta]`.

```bash
python ML/cyclegan_ep/train.py \
  --data-dir data \
  --output-dir outputs/cyclegan/run \
  --model-backend voxel \
  --eta-window-size 11 \
  --phi-window-size 11
```

Inference from a saved checkpoint:

```bash
python ML/cyclegan_ep/infer.py \
  --checkpoint outputs/cyclegan/run/checkpoints/latest.pth \
  --data-dir data \
  --results-dir outputs/cyclegan/inference
```

### Latent image flow

Train the autoencoder first:

```bash
python -m flow.train_autoencoder \
  --data-dir data \
  --output-dir outputs/latent_flow/autoencoder \
  --phi-window-size 11 \
  --eta-window-size 11 \
  --latent-dim 32
```

Then train latent transport using the selected autoencoder checkpoint:

```bash
python -m flow.train_flow \
  --autoencoder-checkpoint outputs/latent_flow/autoencoder/checkpoints/best.pt \
  --output-dir outputs/latent_flow/transport
```

### Shower-shape flow

This pipeline derives shower-shape variables from the ROOT cells and trains a
conditional normalizing flow over those observables.

```bash
python -m flow_electron_to_photon.train \
  --configpath flow_electron_to_photon/configs/config_train.yaml
```

Apply trained fold checkpoints with:

```bash
python -m flow_electron_to_photon.apply \
  --configpath flow_electron_to_photon/configs/config_apply.yaml
```

## Physics plots

Scripts write generated figures to `outputs/plots` by default.

```bash
./scripts/run_basic_plots.sh
./scripts/run_layer2_shower_shapes.sh
./scripts/run_layer2_hotcell_window.sh
./scripts/run_layer2_voxel_ralpha.sh
```

The primary validation order is:

1. Total-energy scale.
2. Mean-window and core-window agreement.
3. `Reta`, `Rphi`, `weta2`, core-energy ratios, and hottest-cell energy.
4. Training losses and latent diagnostics.

## Simulation

The cleaned Geant4 source, electron/photon production macros, build commands,
ROOT schema, and source provenance are documented under `simulation/`. The
primary energy is sampled uniformly from 10 to 50 GeV, which explains why the
nominal `30GeV` and `50GeV` files have nearly identical truth-energy ranges.
Record the seed and software versions for every newly generated production.

## GitHub preparation

Generated outputs, checkpoints, caches, environments, and editor files are
ignored. Before publishing:

```bash
git init
git lfs install
git lfs track "*.root"
git add .
git status
```

Choose and add a license before making the repository public.
