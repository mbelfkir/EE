from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, Tuple

import pandas as pd


class Step(Protocol):
    name: str
    enabled: bool

    def prepare(self, proto: pd.DataFrame, target: pd.DataFrame) -> None: ...
    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
    @property
    def state(self) -> Dict[str, Any]: ...
    @state.setter
    def state(self, state: Dict[str, Any]) -> None: ...
    @property
    def used_variables(self) -> Iterable[str]: ...
