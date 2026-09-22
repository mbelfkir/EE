from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd

from flow_electron_to_photon.processor.protocol import Step


@dataclass
class Processor:
    pre_steps: list[Step]
    post_steps: list[Step]

    def pre(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for step in self.pre_steps:
            if step.enabled:
                proto, target = step.pre(proto, target)
        return proto, target

    def post(self, proto: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for step in self.post_steps:
            if step.enabled:
                proto, target = step.post(proto, target)
        return proto, target

    @property
    def used_variables(self) -> Iterable[str]:
        variables = set()
        for step in [*self.pre_steps, *self.post_steps]:
            variables |= set(step.used_variables)
        return variables
