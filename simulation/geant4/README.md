# Geant4 calorimeter simulation

This directory contains the source of the simplified ATLAS calorimeter used to
produce the electron and photon ROOT samples. The geometry has three
electromagnetic and three hadronic sampling layers. `QGSP_FTFP_BERT` handles
the particle transport and shower development.

## Requirements

- CMake 3.16 or newer.
- A C++ compiler supported by Geant4.
- Geant4 with UI, visualization, and ROOT analysis support.

The source was verified on `belfkir-server` with Geant4 11.4.0, ROOT 6.32.02,
and GCC 13.3.0.

## Build

From the `EE` repository root:

```bash
cmake -S simulation/geant4 -B build/geant4 \
  -DGeant4_DIR=/path/to/geant4/lib/cmake/Geant4
cmake --build build/geant4 -j
```

For a batch-only build, add `-DWITH_GEANT4_UIVIS=OFF`.

## Generate samples

The generator samples the primary energy uniformly between 10 and 50 GeV.
The macros select the particle type and request 10,000 events. Run each sample
in a separate output directory because the simulator writes
`CaloResponse.root` by default.

Electron example:

```bash
mkdir -p outputs/simulation/electron
cd outputs/simulation/electron
../../../build/geant4/CaloR \
  -m ../../../build/geant4/macros/electron_10k.mac \
  -s 12345
mv CaloResponse.root electron_10k.root
```

Photon example:

```bash
mkdir -p outputs/simulation/photon
cd outputs/simulation/photon
../../../build/geant4/CaloR \
  -m ../../../build/geant4/macros/photon_10k.mac \
  -s 23456
mv CaloResponse.root photon_10k.root
```

The cleaned simulator uses the sequential Geant4 run manager. Run independent
jobs in separate directories to parallelize production, and use a distinct
recorded seed for every job.

## Output

The ROOT file contains a `physics` tree with:

- calibrated energy sums for the three ECAL and three HCAL layers;
- `Cal_e` and `Caltotal_e` event-level calorimeter energies in MeV;
- cell energy, position, size, layer, and charged-energy fraction vectors;
- primary energy and momentum in GeV, position in mm, and PDG ID.

Cell energies are retained above the threshold in `include/Constants.hh`.
Detector dimensions, sampling structure, calibration factors, and cell
granularities are defined in the same file.

After generation, validate a sample from the repository root:

```bash
python simulation/inspect_samples.py outputs/simulation/electron
```

## Source layout

- `CaloR.cc`: executable entry point, random seed, threads, and physics list.
- `include/Constants.hh`: geometry, segmentation, and calibration constants.
- `src/DetectorConstruction.cc`: calorimeter materials and geometry.
- `src/PrimaryGeneratorAction.cc`: primary position and energy spectrum.
- `src/EventAction.cc`: cell aggregation and ntuple row filling.
- `src/RunAction.cc`: ROOT tree and branch declaration.
- `macros/`: reproducible electron and photon production commands.

The implementation derives from the Geant4 `Geant4-models` example and keeps
the Geant4 license notices present in the source files.
