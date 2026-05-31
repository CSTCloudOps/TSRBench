from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import numpy as np

from methods import distance_methods as dm

from .base import DistanceMethod, DistanceRegistry
from .utils import cosine_distance, l2_distance

Key = Tuple[int, int]


def _register_classical(registry: DistanceRegistry, anomaly_sort: bool = False) -> None:
    methods = [
        ("Euclidean", dm.euclidean_distance, True),
        ("Manhattan", dm.manhattan_distance, True),
        ("Chebyshev", dm.chebyshev_distance, True),
        ("Pearson_dcor1", partial(dm.pearson_distance, method="dcor1"), True),
        ("Pearson_dcor2", partial(dm.pearson_distance, method="dcor2"), True),
        ("DTW", dm.dtw_distance, False),
        ("STS", dm.STSDistance, False),
        ("STI", dm.sti_distance, False),
        ("SBD", dm.sbd, False),
        ("m-ED", dm.modified_euclidean_distance, True),
        ("m-DTW", dm.modified_dtw_distance, False),
        ("LCSS", dm.lcss_distance, False),
        ("EDR", dm.edr_distance, False),
        ("ERP", dm.erp_distance, False),
        ("SAX", dm.sax_based_distance, True),
        ("DISSIM", dm.dissim_distance, True),
        ("SFA", dm.sfa_based_distance, True),
        ("1DSAX", dm.sax1d_based_distance, True),
    ]
    for name, fn, requires_equal_length in methods:
        registry.register(
            DistanceMethod(
                name=name,
                input_type="series",
                func=fn,
                requires_equal_length=requires_equal_length,
                higher_is_more_anomalous=anomaly_sort,
            )
        )


def _register_embedding_banks(registry: DistanceRegistry, banks: Dict[str, Dict[Key, np.ndarray]]) -> None:
    for bank_name, bank in banks.items():
        registry.register(
            DistanceMethod(
                name=f"{bank_name}_cos",
                input_type="key",
                func=lambda k1, k2, bank=bank: cosine_distance(bank[k1], bank[k2]),
            )
        )
        registry.register(
            DistanceMethod(
                name=f"{bank_name}_l2",
                input_type="key",
                func=lambda k1, k2, bank=bank: l2_distance(bank[k1], bank[k2]),
            )
        )


def build_ucr_r_registry(banks: Dict[str, Dict[Key, np.ndarray]] | None = None) -> DistanceRegistry:
    registry = DistanceRegistry()
    _register_classical(registry, anomaly_sort=False)
    _register_embedding_banks(registry, banks or {})
    return registry


def build_cu_rca_registry() -> DistanceRegistry:
    registry = DistanceRegistry()
    _register_classical(registry, anomaly_sort=True)
    return registry
