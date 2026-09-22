# Shower-shape normalizing flow

This package learns an electron-to-photon correction directly over derived
calorimeter shower-shape variables. It provides ROOT loading, feature building,
preprocessing, fold-based training, checkpoint application, and before/after
physics plots.

The standard feature set contains `reta`, `rphi`, `weta2`, `wstot`, `f1`,
`fside`, `deltae`, and `eratio`. The flow can condition on cluster and layer
energy summaries or on another configured kinematic set.

Train:

```bash
python -m flow_electron_to_photon.train \
  --configpath flow_electron_to_photon/configs/config_train.yaml
```

Apply saved folds:

```bash
python -m flow_electron_to_photon.apply \
  --configpath flow_electron_to_photon/configs/config_apply.yaml
```

Important modules:

- `root_loader.py`: ROOT to pandas loading.
- `physics_features.py`: calorimeter feature construction.
- `processor/steps.py`: preprocessing, training, application, and export.
- `misc/train.py`: conditional flow construction and optimization.
- `plotting/`: distributions, correlations, and loss curves.

The checked-in configs use ROOT files from `data/` and write all generated
artifacts below `outputs/shower_shape_flow/`.
