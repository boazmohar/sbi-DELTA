# File: sbi_delta/simulator/base_simulator.py

import abc
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.filter_bank import FilterBank
from sbi_delta.config import BaseConfig

class BaseSimulator(abc.ABC):
    """
    Abstract base class for SBI‑DELTA simulators.

    Parameters
    ----------
    spectra_manager : SpectraManager
        Must have been initialized with a BaseConfig and .load() called.
    filter_bank : FilterBank
        Must have been initialized with the same BaseConfig.
    config : BaseConfig
        Simulation‐wide config (photon_budget, grid, etc).
    """

    def __init__(self,
                 spectra_manager: SpectraManager,
                 filter_bank: FilterBank,
                 config: BaseConfig):
        self.spectra_manager = spectra_manager
        self.filter_bank = filter_bank
        self.config = config

        # Ensure both pieces share the same grid
        assert hasattr(self.spectra_manager, "wavelength_grid"), "SpectraManager missing grid"
        assert hasattr(self.filter_bank,    "wavelength_grid"), "FilterBank missing grid"
        # They may be different arrays but must match in values:
        if not (self.spectra_manager.wavelength_grid == self.filter_bank.wavelength_grid).all():
            raise ValueError("SpectraManager and FilterBank grids do not match")

    @abc.abstractmethod
    def simulate(self):
        """
        Run the simulation, returning an (n_dyes × n_channels) photon‐count array.
        """
        pass
