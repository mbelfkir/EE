from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import uproot

from flow_electron_to_photon.config import AppConfig, RootDomainSpec
from flow_electron_to_photon.physics_features import EventFeatureComputer


def extract_columns_from_query(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    columns = set()

    class ColumnVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            columns.add(node.id)

        def visit_Call(self, node: ast.Call) -> None:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            parts: list[str] = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                columns.add(".".join(reversed(parts)))

    ColumnVisitor().visit(tree)
    return columns - {"True", "False", "None"}


def configured_raw_branches(cfg: AppConfig, domain_cfg: RootDomainSpec) -> list[str]:
    branches = set()
    for field_name in cfg.root.branches.model_fields:
        value = domain_cfg.branch_overrides.get(field_name, getattr(cfg.root.branches, field_name))
        if value is not None:
            branches.add(value)
    return sorted(branches)


def load_root_domain(
    cfg: AppConfig,
    domain_cfg: RootDomainSpec,
    required_variables: Iterable[str],
    *,
    domain_name: str,
) -> pd.DataFrame:
    builders = cfg.root.physics.feature_builders
    required = sorted(set(required_variables))
    missing = [name for name in required if name not in builders]
    if missing:
        raise KeyError(
            f"Missing feature builders for requested variables: {missing}. "
            "Add them under root.physics.feature_builders."
        )

    raw_branches = configured_raw_branches(cfg, domain_cfg)
    if not raw_branches:
        raise ValueError(
            f"No raw ROOT branches are configured for domain '{domain_name}'. "
            "Check root.branches and branch_overrides in the YAML config."
        )
    rows: list[dict[str, float]] = []
    global_entry = 0

    for file_path in domain_cfg.file_paths:
        tree_path = Path(file_path)
        if not tree_path.exists():
            raise FileNotFoundError(f"ROOT input not found: {tree_path}")

        with uproot.open(tree_path) as root_file:
            if domain_cfg.tree_name not in root_file:
                available = ", ".join(root_file.keys())
                raise KeyError(
                    f"Tree '{domain_cfg.tree_name}' not found in {tree_path}. "
                    f"Available keys: {available}"
                )
            tree = root_file[domain_cfg.tree_name]
            arrays = tree.arrays(raw_branches, library="ak", how=dict)

        n_entries = len(arrays[raw_branches[0]]) if raw_branches else 0
        for local_index in range(n_entries):
            computer = EventFeatureComputer(
                arrays=arrays,
                index=local_index,
                branches=cfg.root.branches,
                physics=cfg.root.physics,
                file_path=str(tree_path),
                entry_index=global_entry,
                domain_name=domain_name,
                branch_overrides=domain_cfg.branch_overrides,
            )
            row = {name: computer.compute(name, builders[name]) for name in required}
            rows.append(row)
            global_entry += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows were loaded for domain '{domain_name}'.")

    nonfinite = {column: int((~np.isfinite(df[column].to_numpy(dtype=np.float64))).sum()) for column in df.columns}
    broken = [name for name, count in nonfinite.items() if count > 0]
    if broken:
        raise ValueError(
            f"Non-finite values were produced for domain '{domain_name}' in columns: {broken}. "
            "Check the branch mappings and feature formulas."
        )

    return df
