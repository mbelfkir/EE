# Simulation

This directory contains the detector-simulation source and the utility used to
audit generated ROOT samples.

## Contents

- `geant4/`: cleaned source for the simplified ATLAS calorimeter simulation.
- `inspect_samples.py`: validates the ROOT tree, branches, units, beam setup,
  energy range, and cell multiplicities.

The Geant4 source was imported from
`mohamed@belfkir-server:/home/mohamed/Geant4-ATLASCalo/ATLAS-simplified` on
2026-09-22. The server repository was on commit
`e694956f92679d8c0943427c02b6f3eb83fd9906`. Its working tree included
`G4AnalysisManager` compatibility changes, which are preserved here, plus an
unfinished multithreading experiment that produced valid output but crashed at
shutdown. The clean copy uses the stable sequential run manager. Build
products, backup snapshots, documentation images, and legacy output-conversion
scripts were not copied.

See [geant4/README.md](geant4/README.md) for build and production commands.

## Verified dataset contract

Observed properties of the four main training files:

- 10,000 events per file.
- Electron PDG ID 11 and photon PDG ID 22.
- Particle source at `x = 0`, `y = 0`, `z = -2069.06 mm`.
- Momentum has `px = 0`, `py = 0`, and positive `pz`.
- Particle energy spans approximately 10-50 GeV in every nominal 30/50 file.
- Mean recorded cells per event are approximately 261 for electrons and 226
  for photons.

Feature-building defaults use these local cell sizes in `(delta eta, delta
phi)`:

- Layer 1: `(0.003125, 0.1)`.
- Layer 2: `(0.025, 0.025)`.
- Layer 3: `(0.05, 0.025)`.

Run `inspect_samples.py` after replacing or regenerating any ROOT file. The
script checks the tree, required branches, particle truth, beam geometry,
energy range, and average cell multiplicity.

The supplied dataset filenames do not encode fixed beam energies. The
generator samples 10-50 GeV uniformly; this is consistent with the audited
`particle_e` distributions.
