from __future__ import annotations

import argparse
from pathlib import Path

from flow_electron_to_photon.main import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and apply the electron-to-photon flow on fold splits.")
    parser.add_argument(
        "--configpath",
        default=str((Path(__file__).resolve().parent / "configs" / "config_train.yaml").resolve()),
        help="Path to the training YAML config.",
    )
    parser.add_argument("--outpath", default=None, help="Optional override for runtime.output_path.")
    args = parser.parse_args()
    run_pipeline(args.configpath, outpath=args.outpath)


if __name__ == "__main__":
    main()
