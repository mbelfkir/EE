from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from flow_electron_to_photon.config import AppConfig
from flow_electron_to_photon.misc.system import setup_output_dir
from flow_electron_to_photon.processor.builder import build_steps
from flow_electron_to_photon.processor.processor import Processor
from flow_electron_to_photon.root_loader import load_root_domain


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_output_path(config_output_path: str, override: str | None = None) -> str:
    base = Path(override if override is not None else config_output_path)
    return str(base if base.is_absolute() else (Path.cwd() / base).resolve())


def run_pipeline(configpath: str, outpath: str | None = None) -> tuple[str, Processor]:
    cfg = AppConfig.load_yaml(configpath)
    set_random_seeds(cfg.runtime.seed)

    output_path = resolve_output_path(cfg.runtime.output_path, override=outpath)
    output_path = setup_output_dir(output_path, allow_recreate=True)

    if cfg.runtime.copy_config:
        shutil.copy2(configpath, Path(output_path) / Path(configpath).name)

    pre_steps, post_steps = build_steps(cfg.processor.pre, cfg.processor.post, output_path)
    processor = Processor(pre_steps=list(pre_steps), post_steps=list(post_steps))

    used_variables = sorted(set(processor.used_variables))
    print(f"\033[1;36m[INFO]\033[92m Loading {len(used_variables)} configured variables from ROOT inputs\033[0m")

    source = load_root_domain(cfg, cfg.io.source, used_variables, domain_name=cfg.io.source_label)
    target = load_root_domain(cfg, cfg.io.target, used_variables, domain_name=cfg.io.target_label)

    print(f"\033[1;36m[INFO]\033[92m Loaded source shape {source.shape}, target shape {target.shape}\033[0m")

    source, target = processor.pre(source, target)
    source, target = processor.post(source, target)
    return output_path, processor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Electron-to-photon shower-shape normalizing-flow pipeline.")
    parser.add_argument("--configpath", required=True, help="Path to the YAML config.")
    parser.add_argument("--outpath", default=None, help="Optional override for runtime.output_path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args.configpath, outpath=args.outpath)


if __name__ == "__main__":
    main()
