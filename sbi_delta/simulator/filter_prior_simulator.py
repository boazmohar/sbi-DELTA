"""Simulator that uses filter parameters from prior."""

from typing import Optional, Sequence
import numpy as np
import torch
from ..simulator.emission_simulator import EmissionSimulator
from ..config import BaseConfig, FilterConfig
from ..spectra_manager import SpectraManager
from ..filter_bank import FilterBank
from ..excitation_manager import ExcitationManager
from ..prior_manager import PriorManager

class FilterPriorSimulator(EmissionSimulator):
    """
    Simulator that uses filter parameters sampled from prior.
    Filter start/stop wavelengths are part of the prior distribution.
    """
    
    def __init__(
        self,
        spectra_manager: SpectraManager,
        config: BaseConfig,
        excitation_manager: ExcitationManager,
        prior_manager: PriorManager,
        n_filters: int = 5,
        max_filter_width: float = 50.0,
    ):
        """
        Initialize simulator with filter parameters in prior.
        
        Args:
            spectra_manager: Manager for fluorophore spectra
            config: Base configuration
            excitation_manager: Manager for excitation wavelengths 
            prior_manager: Manager for sampling parameters
            n_filters: Number of filters to use
            max_filter_width: Maximum allowed filter width in nm
        """
        self.n_filters = n_filters
        self.max_filter_width = max_filter_width
        
        # Create initial filter bank with non-overlapping dummy filters
        filter_cfgs = []
        available_range = config.max_wavelength - config.min_wavelength
        filter_width = available_range / n_filters
        
        for i in range(n_filters):
            start = config.min_wavelength + i * filter_width
            stop = start + filter_width
            filter_cfgs.append(
                FilterConfig(start=start, stop=stop, sharpness=1.0)
            )
        filter_bank = FilterBank(config, filter_cfgs)
        
        super().__init__(
            spectra_manager=spectra_manager,
            filter_bank=filter_bank,
            config=config,
            excitation_manager=excitation_manager,
            prior_manager=prior_manager,
        )
    
    def simulate(self, params=None, add_noise=True, debug=False):
        """
        Simulate measurements using filter parameters from prior.
        
        Args:
            params: Array containing concentrations and filter parameters
            add_noise: Whether to add Poisson noise
            debug: Whether to plot debug information
            
        Returns:
            Array of simulated measurements
        """
        if params is None:
            params = self.prior_manager.sample()
            
        # Split parameters into concentrations and filter params
        n_dyes = len(self.config.dye_names)
        n_bg = 1 if self.config.bg_dye is not None else 0
        n_conc = n_dyes + n_bg
        
        concentrations = params[:n_conc]
        filter_params = params[n_conc:]
        
        # Update filter bank with sampled parameters
        filter_cfgs = []
        for i in range(self.n_filters):
            start = filter_params[i*2]
            width = filter_params[i*2 + 1]
            stop = start + width
            filter_cfgs.append(
                FilterConfig(start=start, stop=stop, sharpness=1.0)
            )
        self.filter_bank = FilterBank(self.config, filter_cfgs)
        
        # Run simulation with updated filters
        return super().simulate(
            concentrations=concentrations,
            add_noise=add_noise,
            debug=debug
        )