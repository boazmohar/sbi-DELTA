# File: sbi_delta/simulator/base_simulator.py

import abc
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.filter_bank import FilterBank
from sbi_delta.config import BaseConfig
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.prior_manager import PriorManager


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
    excitation_manager : ExcitationManager, optional
        Handles excitation spectra and logic.
    prior_manager : PriorManager, optional
        Handles prior distributions for simulation.
    """

    def __init__(self,
                 spectra_manager: SpectraManager,
                 filter_bank: FilterBank,
                 config: BaseConfig,
                 excitation_manager: ExcitationManager = None,
                 prior_manager: PriorManager = None):
        self.spectra_manager = spectra_manager
        self.filter_bank = filter_bank
        self.config = config
        self.excitation_manager = excitation_manager
        self.prior_manager = prior_manager

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
