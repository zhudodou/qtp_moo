"""Spatial multi-objective optimization utilities for QTP renewable-energy siting."""

from .optimization import (
    AlgorithmResult,
    ObjectiveData,
    moead,
    nsga2,
    objective_values,
    spatial_discrimination,
    spea2_with_de,
)

__all__ = [
    "AlgorithmResult",
    "ObjectiveData",
    "moead",
    "nsga2",
    "objective_values",
    "spatial_discrimination",
    "spea2_with_de",
]
