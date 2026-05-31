from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal

InputType = Literal["series", "key"]


@dataclass(frozen=True)
class DistanceMethod:
    name: str
    input_type: InputType
    func: Callable[..., float]
    requires_equal_length: bool = False
    higher_is_more_anomalous: bool = False


class DistanceRegistry:
    def __init__(self) -> None:
        self._methods: Dict[str, DistanceMethod] = {}

    def register(self, method: DistanceMethod) -> None:
        if method.name in self._methods:
            raise KeyError(f"Duplicate distance method: {method.name}")
        self._methods[method.name] = method

    def get(self, name: str) -> DistanceMethod:
        return self._methods[name]

    def names(self) -> list[str]:
        return list(self._methods)

    def items(self):
        return self._methods.items()

