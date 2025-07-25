"""
Configuration classes for the SBI‑DELTA simulator.

The refactoring splits simulation configuration into focused dataclasses:

* :class:`BaseConfig` holds core simulation parameters like the wavelength
  range, wavelength sampling step, photon budget and random seed.  It is
  deliberately simple and does not mix unrelated options.
* :class:`FilterConfig` describes how detection filters are constructed.
  It exposes the filter type (e.g., ``'sigmoid'`` or ``'gaussian'``) and
  parameters controlling filter shape.  For fixed filter simulators it can
  also store the centre wavelengths and bandwidths of each filter.
* :class:`ExcitationConfig` encapsulates options related to modelling the
  excitation process, such as whether to include cross‑talk between fluorophores
  and whether excitation wavelengths are manually specified or searched over a
  range.
* :class:`ParametricFilterConfig` extends :class:`FilterConfig` by adding
  bounds for centre wavelengths and bandwidths.  It is intended for
  parametric filter simulators where filter parameters are part of the
  inference problem.

Each configuration class performs basic validation in ``__post_init__`` to
ensure that the provided values are sensible.  Users can override any of
the defaults when constructing a configuration object.

The classes are implemented as Python dataclasses to provide a concise
declaration of fields and default values.  They include type hints to
facilitate static analysis and editor support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import logging


logger = logging.getLogger(__name__)


def _validate_range(name: str, minimum: float, maximum: float) -> None:
    """Validate that ``minimum`` is strictly less than ``maximum``.

    Parameters
    ----------
    name:
        A human friendly name used in error messages.
    minimum:
        The lower bound of the range.
    maximum:
        The upper bound of the range.

    Raises
    ------
    ValueError
        If ``minimum`` is not strictly less than ``maximum``.
    """
    if minimum >= maximum:
        raise ValueError(f"{name} must satisfy min < max (got {minimum} >= {maximum})")


@dataclass
class BaseConfig:
    """Core simulation parameters.

    This configuration defines the wavelength sampling grid, the photon
    budget for the simulation and the random seed.  All parameters have
    reasonable defaults but can be overridden to suit a particular experiment.
    """

    min_wavelength: float = 350.0
    """Minimum wavelength in nanometres for the simulation grid."""

    max_wavelength: float = 800.0
    """Maximum wavelength in nanometres for the simulation grid."""

    wavelength_step: float = 1.0
    """Step size of the wavelength sampling grid in nanometres."""

    photon_budget: float = 1e5
    """Total number of photons allocated across all fluorophores and channels."""

    random_seed: Optional[int] = None
    """Seed for the random number generator used within the simulator."""

    def __post_init__(self) -> None:
        logger.debug("Initialising BaseConfig with values: %s", self)
        _validate_range("wavelength range", self.min_wavelength, self.max_wavelength)
        if self.wavelength_step <= 0:
            raise ValueError(f"wavelength_step must be positive (got {self.wavelength_step})")
        if self.photon_budget <= 0:
            raise ValueError(f"photon_budget must be positive (got {self.photon_budget})")


@dataclass
class FilterConfig:
    """Configuration for detection filters using start, stop, and sharpness.

    Parameters
    ----------
    start:
        The start wavelength (nm) of the filter's passband.
    stop:
        The stop wavelength (nm) of the filter's passband.
    sharpness:
        Controls how quickly the filter transitions from pass to stop band.
        Must be positive.
    """

    start: float = 400.0
    stop: float = 700.0
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        logger.debug("Initialising FilterConfig with values: %s", self)
        if self.start >= self.stop:
            raise ValueError(f"start must be less than stop (got {self.start} >= {self.stop})")
        if self.sharpness <= 0:
            raise ValueError(f"sharpness must be positive (got {self.sharpness})")



@dataclass
class ExcitationConfig:
    """Configuration for modelling excitation and cross‑talk.

    Parameters
    ----------
    include_crosstalk:
        Whether to model cross‑talk between fluorophores.  When ``False``
        excitation spectra are assumed to be independent.
    use_manual_wavelengths:
        If ``True``, the user provides explicit excitation wavelengths via
        :attr:`manual_wavelengths`.  Otherwise the simulator may determine
        excitation wavelengths itself or search over a range.
    manual_wavelengths:
        Explicit excitation wavelengths to use when :attr:`use_manual_wavelengths`
        is ``True``.  If provided, the length should match the number of
        fluorophores.  If ``None``, the simulator chooses default values.
    search_range:
        A tuple ``(min, max)`` specifying the range over which excitation
        wavelengths may be sampled when not using manual wavelengths.  If
        provided, ``min`` must be strictly less than ``max``.
    """

    include_crosstalk: bool = True
    use_manual_wavelengths: bool = False
    manual_wavelengths: Optional[Sequence[float]] = None
    search_range: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        logger.debug("Initialising ExcitationConfig with values: %s", self)
        if self.use_manual_wavelengths:
            if self.manual_wavelengths is None:
                raise ValueError(
                    "manual_wavelengths must be provided when use_manual_wavelengths is True"
                )
        if self.manual_wavelengths is not None:
            if not self.use_manual_wavelengths:
                logger.warning(
                    "manual_wavelengths provided but use_manual_wavelengths is False; values will be ignored"
                )
            # ensure all wavelengths are positive
            for wl in self.manual_wavelengths:
                if wl <= 0:
                    raise ValueError(f"manual_wavelengths must be positive (got {wl})")
        if self.search_range is not None:
            min_wl, max_wl = self.search_range
            _validate_range("search_range", min_wl, max_wl)