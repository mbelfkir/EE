from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import uproot


def setup_output_dir(outpath: str, allow_recreate: bool = False) -> str:
    try:
        os.makedirs(outpath, exist_ok=allow_recreate)
    except Exception as exc:
        print(f"\033[1;91m[ERROR] ❌ Cannot create or open directory for results.\033[0m \n Details: {exc}")
        raise exc from None

    print(f"\033[1;36m[INFO]\033[92m Output directory is:\033[0m {outpath}")
    return outpath


def save_stats_hdf5(mean: pd.Series, std: pd.Series, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("variables", data=np.asarray(mean.index.astype(str).tolist(), dtype="S256"))
        handle.create_dataset("mean", data=mean.to_numpy(dtype=np.float64))
        handle.create_dataset("std", data=std.to_numpy(dtype=np.float64))


def load_stats_hdf5(path: str | Path) -> tuple[pd.Series, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Statistics file not found: {path}")
    with h5py.File(path, "r") as handle:
        variables = [value.decode("utf-8") for value in handle["variables"][...]]
        mean = pd.Series(handle["mean"][...], index=variables, dtype=np.float64)
        std = pd.Series(handle["std"][...], index=variables, dtype=np.float64)
    return mean, std


def export_numeric_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "hdf5") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_fmt = str(fmt).lower()

    if normalized_fmt == "hdf5":
        with h5py.File(output_path, "w") as handle:
            for column in df.columns:
                values = df[column].to_numpy()
                if values.dtype.kind not in {"b", "i", "u", "f"}:
                    raise TypeError(
                        f"HDF5 export only supports numeric columns; column '{column}' has dtype {values.dtype}."
                    )
                handle.create_dataset(column, data=values)
        return output_path

    if normalized_fmt == "parquet":
        try:
            df.to_parquet(output_path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Parquet export failed. Install a parquet backend such as pyarrow, or use HDF5 export."
            ) from exc
        return output_path

    if normalized_fmt == "root":
        payload: dict[str, np.ndarray] = {}
        for column in df.columns:
            values = df[column].to_numpy()
            if values.dtype.kind not in {"b", "i", "u", "f"}:
                raise TypeError(
                    f"ROOT export only supports numeric columns; column '{column}' has dtype {values.dtype}."
                )
            payload[column] = values
        with uproot.recreate(output_path) as handle:
            handle["table"] = payload
        return output_path

    raise ValueError(f"Unsupported export format: {fmt}")
