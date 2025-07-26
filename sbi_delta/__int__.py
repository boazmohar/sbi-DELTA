"""
sbi_delta package
~~~~~~~~~~~~~~~~~~

This package contains refactored components of the SBI‑DELTA simulator.

Initially the package exposes only configuration classes.  Future versions
will include a spectra manager, filter bank and simulator variants as
described in the refactoring plan.

The configuration classes are defined in :mod:`sbi_delta.config`.
"""

from .config import (
    BaseConfig,
    FilterConfig,
    ExcitationConfig,
    ParametricFilterConfig,
)

from .spectra_manager import SpectraManager 
from .filter_bank import FilterBank 

__all__ = [
    "BaseConfig",
    "FilterConfig",
    "ExcitationConfig",
    "ParametricFilterConfig",
]