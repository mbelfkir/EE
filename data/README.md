# Dataset

The files in this directory are ROOT ntuples with a `physics` tree.

## Files

| File | Entries | Particle |
| --- | ---: | --- |
| `electron_30GeV_10k.root` | 10,000 | electron, PDG 11 |
| `electron_50GeV_10k.root` | 10,000 | electron, PDG 11 |
| `photon_10GeV_10k.root` | 10,000 | photon, PDG 22 |
| `photon_30GeV_10k.root` | 10,000 | photon, PDG 22 |
| `photon_50GeV_10k.root` | 10,000 | photon, PDG 22 |

The main training configs use the electron and photon files carrying nominal
`30GeV` and `50GeV` labels. The optional photon `10GeV` sample is retained as
part of the supplied dataset but is not selected by those configs.

## Important naming caveat

The filename energy is not a fixed truth energy. Auditing `particle_e` in the
four main files gives ranges close to 10-50 GeV and means close to 30 GeV in
every file. Code that parses `30GeV` or `50GeV` from the filename uses a nominal
sample label. It should not use that value as event truth.

Run the included inspection tool to regenerate the audit:

```bash
python simulation/inspect_samples.py data --output outputs/dataset_summary.json
```

## Main branches

- Cell data: `cell_e`, `cell_x`, `cell_y`, `cell_dx`, `cell_dy`, `cell_l`.
- ECAL summaries: `ECAL1_e`, `ECAL2_e`, `ECAL3_e`.
- HCAL summaries: `HCAL1_e`, `HCAL2_e`, `HCAL3_e`.
- Event summaries: `Cal_e`, `Caltotal_e`.
- Particle truth: `particle_e`, position, momentum, and `particle_pdgId`.

The current feature builders convert local cell coordinates with
`eta = cell_x / 1440` and `phi = cell_y / 1440`. Layer-2 image windows are
centered on the highest-energy layer-2 cell.

## Provenance still required

Before public physics use, document the simulation software and version,
detector geometry version, physics configuration, generation spectrum, random
seeds, and the command that produced each file.
