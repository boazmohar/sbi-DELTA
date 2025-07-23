"""
Enhanced SBI Simulator module with filter parameters as part of the parameter vector.

This module extends the SBI simulator to include filter characteristics (center wavelengths
and bandwidths) as part of the parameter vector that gets sampled from priors, rather than
being fixed hyperparameters.
"""

import numpy as np
import torch
from torch.distributions import Dirichlet, Beta, Uniform, Distribution, Independent
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
from scipy.interpolate import interp1d
import warnings
from dataclasses import dataclass, field

from .Microscope import find_optimal_excitation
from .io import list_fluorophores
from .sbi_simulator import SBIConfig, FilterBank, SpectraManager


@dataclass
class EnhancedSBIConfig(SBIConfig):
    """Enhanced configuration for SBI simulation parameters including filter priors."""
    
    # Filter parameter priors
    center_wavelength_bounds: Tuple[float, float] = (500, 800)
    bandwidth_bounds: Tuple[float, float] = (10, 50)
    min_wavelength_separation: float = 10.0
    center_wavelength_distribution: str = 'uniform'  # Options: 'uniform', 'peak_centered', 'tile_peaks'
    peak_centered_std: float = 10.0
    
    # Parameter indexing
    n_channels: int = 5  # Number of detection channels
    include_filter_params: bool = True  # Whether to include filter params in theta
    include_background_params: bool = False
    background_bounds: Tuple[float, float] = (1.0, 200.0)


class CustomFlatPrior(Distribution):
    """
    Custom flat prior for SBI parameters combining concentration, filter center wavelengths,
    filter bandwidths, and background parameters.
    
    This prior combines multiple independent distributions into a single joint prior
    while maintaining a flat log probability for SBI compatibility.
    """
    
    def __init__(
        self,
        n_fluorophores: int,
        n_channels: int,
        concentration_prior: Distribution,
        center_prior: Distribution,
        bandwidth_prior: Distribution,
        background_prior: Optional[Distribution] = None,
        include_background: bool = True
    ):
        """
        Initialize the custom flat prior.
        
        Args:
            n_fluorophores: Number of fluorophore concentration parameters
            n_channels: Number of detection channels
            concentration_prior: Prior for concentration parameters
            center_prior: Prior for filter center wavelengths
            bandwidth_prior: Prior for filter bandwidths
            background_prior: Prior for background parameters (optional)
            include_background: Whether to include background parameters
        """
        super().__init__()
        
        self.n_fluorophores = n_fluorophores
        self.n_channels = n_channels
        self.include_background = include_background
        
        # Store individual priors
        self.concentration_prior = concentration_prior
        self.center_prior = center_prior
        self.bandwidth_prior = bandwidth_prior
        self.background_prior = background_prior
        
        # Calculate parameter dimensions
        self.n_concentration_params = n_fluorophores
        self.n_center_params = n_channels
        self.n_bandwidth_params = n_channels
        self.n_background_params = 1 if include_background else 0
        
        self.total_params = (
            self.n_concentration_params + 
            self.n_center_params + 
            self.n_bandwidth_params + 
            self.n_background_params
        )
        
        # Create parameter slices for easy access
        self.concentration_slice = slice(0, self.n_concentration_params)
        self.center_slice = slice(
            self.n_concentration_params,
            self.n_concentration_params + self.n_center_params
        )
        self.bandwidth_slice = slice(
            self.n_concentration_params + self.n_center_params,
            self.n_concentration_params + self.n_center_params + self.n_bandwidth_params
        )
        if self.include_background:
            self.background_slice = slice(
                self.n_concentration_params + self.n_center_params + self.n_bandwidth_params,
                self.total_params
            )
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the joint prior.
        
        Args:
            sample_shape: Shape of samples to generate
            
        Returns:
            Tensor of shape (*sample_shape, total_params) with samples from the prior
        """
        samples = []
        
        # Sample concentration parameters
        if isinstance(self.concentration_prior, Dirichlet):
            concentration_samples = self.concentration_prior.sample(sample_shape)
        else:
            # Handle independent distributions
            concentration_samples = torch.stack([
                self.concentration_prior.sample(sample_shape) 
                for _ in range(self.n_concentration_params)
            ], dim=-1)
        samples.append(concentration_samples)
        
        # Sample center wavelengths
        if isinstance(self.center_prior, torch.distributions.Normal):
            center_samples = self.center_prior.sample(sample_shape)
        else:
            center_samples = torch.stack([
                self.center_prior.sample(sample_shape) 
                for _ in range(self.n_center_params)
            ], dim=-1)
        samples.append(center_samples)
        
        # Sample bandwidths
        bandwidth_samples = torch.stack([
            self.bandwidth_prior.sample(sample_shape) 
            for _ in range(self.n_bandwidth_params)
        ], dim=-1)
        samples.append(bandwidth_samples)
        
        # Sample background parameters
        if self.include_background and self.background_prior is not None:
            background_samples = self.background_prior.sample(sample_shape)
            if len(background_samples.shape) == len(sample_shape):
                background_samples = background_samples.unsqueeze(-1)
            samples.append(background_samples)
        
        # Concatenate all samples
        return torch.cat(samples, dim=-1)
    
    def log_prob(self, theta):
        """
        Return flat log probability for SBI compatibility.
        
        Args:
            theta: Parameter tensor
            
        Returns:
            Zero tensor with appropriate shape
        """
        return torch.zeros(theta.shape[:-1])
    
    @property
    def support(self):
        """Disable internal constraint checks."""
        raise NotImplementedError("Support check bypassed for custom prior.")
    
    def extract_parameters(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract individual parameter groups from joint parameter vector.
        
        Args:
            theta: Joint parameter tensor
            
        Returns:
            Dictionary with separated parameter groups
        """
        params = {
            'concentrations': theta[..., self.concentration_slice],
            'center_wavelengths': theta[..., self.center_slice],
            'bandwidths': theta[..., self.bandwidth_slice],
        }
        
        if self.include_background:
            params['background'] = theta[..., self.background_slice]
        
        return params


class EnhancedSBISimulator:
    """
    Enhanced simulator class that includes filter parameters in the parameter vector.
    """
    
    def __init__(
        self, 
        fluorophore_names: List[str],
        spectra_dir: Union[str, Path],
        config: Optional[EnhancedSBIConfig] = None
    ):
        self.fluorophore_names = fluorophore_names
        self.config = config or EnhancedSBIConfig()
        self.n_fluorophores = len(fluorophore_names)
        
        # Initialize components from original simulator
        self.spectra_manager = SpectraManager(spectra_dir, self.config)
        self.filter_bank = FilterBank(self.config)
        
        # Load spectra
        self.emission_spectra = self.spectra_manager.load_and_interpolate_emission(fluorophore_names)
        self.excitation_spectra = self.spectra_manager.load_and_interpolate_excitation(fluorophore_names)
        self.background_spectrum = self.spectra_manager.load_background_spectrum(
            self.config.background_fluorophore
        )
        
        # Validate that all spectra were loaded
        missing = set(fluorophore_names) - set(self.emission_spectra.keys())
        if missing:
            raise ValueError(f"Missing emission spectra for fluorophores: {missing}")
        
        # Set up excitation wavelengths (same as original)
        self.excitation_wavelengths = None
        if self.config.include_excitation_crosstalk:
            if self.config.excitation_wavelengths is not None:
                self.excitation_wavelengths = self.config.excitation_wavelengths
            elif self.config.optimize_excitation and self.excitation_spectra:
                self.excitation_wavelengths = self._optimize_excitation_wavelengths()
        
        # Set up random number generator
        self.rng = np.random.default_rng(seed=self.config.random_seed)
        
        # Load background excitation spectrum if available
        self.background_excitation = None
        if self.config.include_excitation_crosstalk:
            try:
                bg_excitation = self.spectra_manager.load_and_interpolate_excitation([self.config.background_fluorophore])
                if self.config.background_fluorophore in bg_excitation:
                    self.background_excitation = bg_excitation[self.config.background_fluorophore]
            except Exception as e:
                warnings.warn(f"Could not load background excitation spectrum: {e}")
        
        # Calculate parameter dimensions
        self._calculate_parameter_dimensions()
    
    def _calculate_parameter_dimensions(self):
        """Calculate the dimensions of the parameter vector."""
        self.n_concentration_params = self.n_fluorophores
        
        if self.config.include_filter_params:
            self.n_filter_params = 2 * self.config.n_channels  # centers + bandwidths
        else:
            self.n_filter_params = 0

        if self.config.include_background_params:
            self.n_background_params = 1
        else:
            self.n_background_params = 0
            
        self.total_params = self.n_concentration_params + self.n_filter_params + self.n_background_params
        
        # Create parameter indices for easy access
        self.concentration_slice = slice(0, self.n_concentration_params)
        
        last_idx = self.n_concentration_params
        if self.config.include_filter_params:
            self.center_slice = slice(
                last_idx, 
                last_idx + self.config.n_channels
            )
            self.bandwidth_slice = slice(
                last_idx + self.config.n_channels,
                last_idx + 2 * self.config.n_channels
            )
            last_idx += self.n_filter_params

        if self.config.include_background_params:
            self.background_slice = slice(last_idx, last_idx + self.n_background_params)
    
    def _optimize_excitation_wavelengths(self) -> List[float]:
        """
        Optimize excitation wavelengths for the loaded fluorophores.
        """
        try:
            # Try advanced optimization first
            from .advanced_optimization import find_optimal_excitation_advanced, OptimizationConfig
            
            # Create optimization config
            opt_config = OptimizationConfig(
                search_range=self.config.excitation_search_range,
                min_wavelength_separation=10.0,
                n_multistart=5,
                population_size=15,
                max_iterations=500
            )
            
            optimal_dict = find_optimal_excitation_advanced(
                self.fluorophore_names,
                self.spectra_manager.spectra_dir,
                config=opt_config
            )
            return [optimal_dict[name] for name in self.fluorophore_names]
            
        except Exception as e:
            warnings.warn(f"Advanced optimization failed: {e}. Trying basic optimization.")
            
            try:
                # Fallback to basic optimization
                from .Microscope import find_optimal_excitation
                
                optimal_dict = find_optimal_excitation(
                    self.fluorophore_names,
                    self.spectra_manager.spectra_dir,
                    search_range=self.config.excitation_search_range
                )
                return [optimal_dict[name] for name in self.fluorophore_names]
                
            except Exception as e2:
                warnings.warn(f"Basic optimization also failed: {e2}. Using peak excitation wavelengths.")
                return self._get_peak_excitation_wavelengths()
    
    def _get_peak_excitation_wavelengths(self) -> List[float]:
        """Get peak excitation wavelengths for each fluorophore."""
        peak_wavelengths = []
        for name in self.fluorophore_names:
            if name in self.excitation_spectra:
                spectrum = self.excitation_spectra[name]
                peak_idx = np.argmax(spectrum)
                peak_wl = self.spectra_manager.wavelengths[peak_idx]
                peak_wavelengths.append(peak_wl)
            else:
                peak_wavelengths.append(500.0)
                warnings.warn(f"No excitation spectrum for {name}, using default 500nm")
        
        return peak_wavelengths

    def _get_peak_emission_wavelengths(self) -> List[float]:
        """Get peak emission wavelengths for each fluorophore."""
        peak_wavelengths = []
        for name in self.fluorophore_names:
            if name in self.emission_spectra:
                spectrum = self.emission_spectra[name]
                peak_idx = np.argmax(spectrum)
                peak_wl = self.spectra_manager.wavelengths[peak_idx]
                peak_wavelengths.append(peak_wl)
            else:
                # Fallback to a reasonable default if emission spectrum is missing
                peak_wavelengths.append(600.0)
                warnings.warn(f"No emission spectrum for {name}, using default 600nm")
        return peak_wavelengths
    
    def calculate_excitation_crosstalk_matrix(
        self, 
        excitation_wavelengths: Optional[List[float]] = None
    ) -> np.ndarray:
        """Calculate excitation crosstalk matrix (same as original)."""
        if excitation_wavelengths is None:
            excitation_wavelengths = self.excitation_wavelengths
        
        if excitation_wavelengths is None:
            n_fluors = len(self.fluorophore_names)
            return np.eye(n_fluors)
        
        n_fluors = len(self.fluorophore_names)
        crosstalk_matrix = np.zeros((n_fluors, n_fluors))
        
        for i, exc_wl in enumerate(excitation_wavelengths):
            for j, fluor_name in enumerate(self.fluorophore_names):
                if fluor_name in self.excitation_spectra:
                    wl_idx = np.argmin(np.abs(self.spectra_manager.wavelengths - exc_wl))
                    crosstalk_matrix[i, j] = self.excitation_spectra[fluor_name][wl_idx]
                else:
                    crosstalk_matrix[i, j] = 1.0 if i == j else 0.0
        
        return crosstalk_matrix
    
    def calculate_background_excitation_response(
        self,
        excitation_wavelengths: Optional[List[float]] = None,
        laser_powers: Optional[np.ndarray] = None
    ) -> float:
        """Calculate background fluorescence response to excitation lasers."""
        if self.background_excitation is None or excitation_wavelengths is None:
            return 1.0
        
        if laser_powers is None:
            laser_powers = np.ones(len(excitation_wavelengths))
        
        total_bg_excitation = 0.0
        for exc_wl, power in zip(excitation_wavelengths, laser_powers):
            wl_idx = np.argmin(np.abs(self.spectra_manager.wavelengths - exc_wl))
            bg_response = self.background_excitation[wl_idx]
            total_bg_excitation += power * bg_response
        
        return total_bg_excitation
    
    def extract_parameters(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract concentration and filter parameters from parameter vector.
        
        Args:
            theta: Parameter vector of shape (batch_size, total_params)
            
        Returns:
            Dictionary with 'concentrations', 'center_wavelengths', 'bandwidths'
        """
        if len(theta.shape) == 1:
            theta = theta[np.newaxis, :]
            
        params = {'concentrations': theta[:, self.concentration_slice]}
        
        if self.config.include_filter_params:
            params['center_wavelengths'] = theta[:, self.center_slice]
            params['bandwidths'] = theta[:, self.bandwidth_slice]
        else:
            # Use default values if filter params not included
            params['center_wavelengths'] = np.full((theta.shape[0], self.config.n_channels), 600.0)
            params['bandwidths'] = np.full((theta.shape[0], self.config.n_channels), 20.0)

        if self.config.include_background_params:
            params['background'] = theta[:, self.background_slice]
        
        return params
    
    def simulate_batch(
        self,
        theta: np.ndarray,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulate detected photon counts for a batch of parameter combinations.
        
        Args:
            theta: Parameter vector of shape (batch_size, total_params)
            add_noise: Whether to add Poisson noise
            
        Returns:
            Tensor of shape (batch_size, n_channels) with detected photon counts
        """
        params = self.extract_parameters(theta)
        concentrations = params['concentrations']
        center_wavelengths = params['center_wavelengths']
        bandwidths = params['bandwidths']
        background = params.get('background')
        
        batch_size = theta.shape[0]
        n_channels = self.config.n_channels
        
        results = []
        
        for batch_idx in range(batch_size):
            # Extract parameters for this sample
            conc = concentrations[batch_idx]
            centers = center_wavelengths[batch_idx]
            bw = bandwidths[batch_idx]
            
            # Create detection filters for this sample
            filters = self.filter_bank.create_filters(centers.tolist(), bw.tolist())
            
            # Prepare emission spectra matrix
            emission_matrix = np.stack([
                self.emission_spectra[name] for name in self.fluorophore_names
            ])  # Shape: (n_fluorophores, n_wavelengths)
            
            # Calculate dye signals for each channel
            dye_signals = np.zeros(n_channels)
            for i, fluorophore_conc in enumerate(conc):
                if fluorophore_conc > 0:
                    spectrum = emission_matrix[i]
                    channel_responses = np.sum(filters * spectrum, axis=1)
                    dye_signals += fluorophore_conc * channel_responses
            
            # Normalize dye signals to photon budget
            total_dye_signal = dye_signals.sum()
            if total_dye_signal > 0:
                dye_photons = dye_signals / total_dye_signal * self.config.total_dye_photons
            else:
                dye_photons = np.zeros(n_channels)
            
            # Add background signal
            background_responses = np.sum(filters * self.background_spectrum, axis=1)
            background_total = background_responses.sum()
            if background_total > 0:
                total_background_photons = (
                    background[batch_idx, 0]
                    if background is not None
                    else self.config.total_background_photons
                )
                background_photons = (background_responses / background_total * 
                                    total_background_photons)
            else:
                background_photons = np.zeros(n_channels)
            
            # Total signal
            total_signal = dye_photons + background_photons
            
            # Add Poisson noise if requested
            if add_noise:
                detected_counts = self.rng.poisson(total_signal).astype(np.float32)
            else:
                detected_counts = total_signal.astype(np.float32)
            
            results.append(detected_counts)
        
        return torch.tensor(results, dtype=torch.float32)
    
    def simulate_batch_with_excitation(
        self,
        theta: np.ndarray,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulate detected photon counts with excitation crosstalk modeling.
        
        Args:
            theta: Parameter vector of shape (batch_size, total_params)
            add_noise: Whether to add Poisson noise
            
        Returns:
            Tensor of shape (batch_size, n_channels) with detected photon counts
        """
        if not self.config.include_excitation_crosstalk:
            return self.simulate_batch(theta, add_noise)
        
        params = self.extract_parameters(theta)
        concentrations = params['concentrations']
        center_wavelengths = params['center_wavelengths']
        bandwidths = params['bandwidths']
        background = params.get('background')
        
        batch_size = theta.shape[0]
        n_channels = self.config.n_channels
        
        results = []
        
        for batch_idx in range(batch_size):
            # Extract parameters for this sample
            conc = concentrations[batch_idx]
            centers = center_wavelengths[batch_idx]
            bw = bandwidths[batch_idx]
            
            # Create detection filters for this sample
            filters = self.filter_bank.create_filters(centers.tolist(), bw.tolist())
            
            # Calculate excitation crosstalk matrix
            crosstalk_matrix = self.calculate_excitation_crosstalk_matrix(self.excitation_wavelengths)
            
            # Prepare emission spectra matrix
            emission_matrix = np.stack([
                self.emission_spectra[name] for name in self.fluorophore_names
            ])  # Shape: (n_fluorophores, n_wavelengths)
            
            # Apply excitation crosstalk
            excited_amplitudes = np.zeros(self.n_fluorophores)
            for i, laser_power in enumerate(conc):
                if laser_power > 0:
                    excited_amplitudes += laser_power * crosstalk_matrix[i, :]
            
            # Calculate dye signals for each channel
            dye_signals = np.zeros(n_channels)
            for i, excited_amp in enumerate(excited_amplitudes):
                if excited_amp > 0:
                    spectrum = emission_matrix[i]
                    channel_responses = np.sum(filters * spectrum, axis=1)
                    dye_signals += excited_amp * channel_responses
            
            # Normalize dye signals to photon budget
            total_dye_signal = dye_signals.sum()
            if total_dye_signal > 0:
                dye_photons = dye_signals / total_dye_signal * self.config.total_dye_photons
            else:
                dye_photons = np.zeros(n_channels)
            
            # Add background signal with excitation-dependent response
            bg_excitation_response = self.calculate_background_excitation_response(
                self.excitation_wavelengths, conc
            )
            background_responses = np.sum(filters * self.background_spectrum, axis=1)
            background_total = background_responses.sum()
            if background_total > 0:
                total_background_photons = (
                    background[batch_idx, 0]
                    if background is not None
                    else self.config.total_background_photons
                )
                background_photons = (background_responses / background_total * 
                                    total_background_photons * bg_excitation_response)
            else:
                background_photons = np.zeros(n_channels)
            
            # Total signal
            total_signal = dye_photons + background_photons
            
            # Add Poisson noise if requested
            if add_noise:
                detected_counts = self.rng.poisson(total_signal).astype(np.float32)
            else:
                detected_counts = total_signal.astype(np.float32)
            
            results.append(detected_counts)
        
        return torch.tensor(results, dtype=torch.float32)
    
    def create_dirichlet_prior(self, concentration: float = 1.0) -> Dirichlet:
        """Create Dirichlet prior for concentration ratios."""
        return Dirichlet(concentration * torch.ones(self.n_fluorophores))
    
    def create_beta_prior(self, alpha: float = 1.0, beta: float = 1.0) -> Beta:
        """Create Beta prior for individual concentrations."""
        return Beta(alpha, beta)
    
    def create_uniform_prior(self, low: float, high: float) -> Uniform:
        """Create Uniform prior for filter parameters."""
        return Uniform(low, high)
    
    def create_custom_prior(
        self,
        concentration_prior: Optional[Distribution] = None,
        center_prior: Optional[Distribution] = None,
        bandwidth_prior: Optional[Distribution] = None,
        background_prior: Optional[Distribution] = None,
        prior_config: Dict = None
    ) -> CustomFlatPrior:
        """
        Create custom flat prior for all parameters.
        
        Args:
            concentration_prior: Prior for concentration parameters
            center_prior: Prior for filter center wavelengths
            bandwidth_prior: Prior for filter bandwidths
            background_prior: Prior for background parameters
            prior_config: Configuration dictionary for default priors
            
        Returns:
            CustomFlatPrior instance
        """
        if prior_config is None:
            prior_config = {}
        
        # Default concentration prior (Dirichlet)
        if concentration_prior is None:
            concentration = prior_config.get('concentration', 1.0)
            concentration_prior = self.create_dirichlet_prior(concentration)
        
        # Default center wavelength prior
        if center_prior is None:
            if self.config.center_wavelength_distribution == 'uniform':
                center_low = prior_config.get('center_low', self.config.center_wavelength_bounds[0])
                center_high = prior_config.get('center_high', self.config.center_wavelength_bounds[1])
                center_prior = self.create_uniform_prior(center_low, center_high)
            elif self.config.center_wavelength_distribution == 'peak_centered':
                peak_wavelengths = self._get_peak_emission_wavelengths()
                
                if len(peak_wavelengths) != self.config.n_channels:
                    warnings.warn(
                        f"Number of fluorophores ({len(peak_wavelengths)}) does not match "
                        f"number of channels ({self.config.n_channels}). "
                        "Using uniform distribution for center wavelengths."
                    )
                    center_low = prior_config.get('center_low', self.config.center_wavelength_bounds[0])
                    center_high = prior_config.get('center_high', self.config.center_wavelength_bounds[1])
                    center_prior = self.create_uniform_prior(center_low, center_high)
                else:
                    locs = torch.tensor(peak_wavelengths, dtype=torch.float32)
                    scales = torch.full_like(locs, self.config.peak_centered_std)
                    center_prior = torch.distributions.Normal(locs, scales)
            
            elif self.config.center_wavelength_distribution == 'tile_peaks':
                peak_wavelengths = self._get_peak_emission_wavelengths()
                
                if not peak_wavelengths:
                    warnings.warn("No peak emission wavelengths found. Using uniform distribution.")
                    center_low = prior_config.get('center_low', self.config.center_wavelength_bounds[0])
                    center_high = prior_config.get('center_high', self.config.center_wavelength_bounds[1])
                    center_prior = self.create_uniform_prior(center_low, center_high)
                else:
                    min_peak = min(peak_wavelengths)
                    max_peak = max(peak_wavelengths)
                    
                    # Create centers that tile the space between min and max peaks
                    tiled_centers = torch.linspace(min_peak, max_peak, self.config.n_channels)
                    
                    locs = tiled_centers.float()
                    scales = torch.full_like(locs, self.config.peak_centered_std)
                    center_prior = torch.distributions.Normal(locs, scales)
            else:
                raise ValueError(
                    f"Invalid center_wavelength_distribution: {self.config.center_wavelength_distribution}"
                )
        
        # Default bandwidth prior (Uniform)
        if bandwidth_prior is None:
            bw_low = prior_config.get('bandwidth_low', self.config.bandwidth_bounds[0])
            bw_high = prior_config.get('bandwidth_high', self.config.bandwidth_bounds[1])
            bandwidth_prior = self.create_uniform_prior(bw_low, bw_high)
        
        # Default background prior (Uniform)
        if background_prior is None:
            bg_low = prior_config.get('background_low', 10.0)
            bg_high = prior_config.get('background_high', 100.0)
            background_prior = self.create_uniform_prior(bg_low, bg_high)
        
        return CustomFlatPrior(
            n_fluorophores=self.n_fluorophores,
            n_channels=self.config.n_channels,
            concentration_prior=concentration_prior,
            center_prior=center_prior,
            bandwidth_prior=bandwidth_prior,
            background_prior=background_prior,
            include_background=self.config.include_background_params
        )
    
    def create_prior(self, prior_config: Dict = None) -> torch.distributions.Distribution:
        """
        Create joint prior for all parameters including filter parameters.
        
        Args:
            prior_config: Configuration for prior distributions
            
        Returns:
            Joint prior distribution
        """
        if prior_config is None:
            prior_config = {}
        
        # Default prior configuration
        concentration_prior_type = prior_config.get('concentration_prior_type', 'dirichlet')
        concentration_params = prior_config.get('concentration_params', {'concentration': 1.0})
        
        center_params = prior_config.get('center_params', {
            'low': self.config.center_wavelength_bounds[0],
            'high': self.config.center_wavelength_bounds[1]
        })
        
        bandwidth_params = prior_config.get('bandwidth_params', {
            'low': self.config.bandwidth_bounds[0],
            'high': self.config.bandwidth_bounds[1]
        })
        
        # Create independent priors for each parameter type
        priors = []
        
        # Concentration priors
        if concentration_prior_type == 'dirichlet':
            concentration_prior = self.create_dirichlet_prior(
                concentration_params.get('concentration', 1.0)
            )
            priors.append(concentration_prior)
        elif concentration_prior_type == 'beta':
            alpha = concentration_params.get('alpha', 1.0)
            beta = concentration_params.get('beta', 1.0)
            beta_prior = self.create_beta_prior(alpha, beta)
            # Independent Beta for each fluorophore
            priors.extend([beta_prior] * self.n_fluorophores)
        
        # Filter parameter priors
        if self.config.include_filter_params:
            # Center wavelengths
            center_prior = self.create_uniform_prior(
                center_params['low'], center_params['high']
            )
            priors.extend([center_prior] * self.config.n_channels)
            
            # Bandwidths
            bandwidth_prior = self.create_uniform_prior(
                bandwidth_params['low'], bandwidth_params['high']
            )
            priors.extend([bandwidth_prior] * self.config.n_channels)
        
        # Combine into joint prior
        from torch.distributions import Independent
        
        # Create independent distributions for each parameter
        joint_prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(self.total_params),
                torch.ones(self.total_params)
            ),
            1
        )
        
        # Note: This is a placeholder. In practice, you might want to use
        # torch.distributions.MixtureSameFamily or implement a custom distribution
        # For now, we'll return the individual priors for sampling
        
        return priors
    
    def generate_training_data(
        self,
        n_samples: int,
        prior_config: Dict = None,
        add_noise: bool = True,
        use_custom_prior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for SBI with filter parameters included.
        
        Args:
            n_samples: Number of training samples
            prior_config: Configuration for prior distributions
            add_noise: Whether to add Poisson noise
            use_custom_prior: Whether to use CustomFlatPrior instead of individual priors
            
        Returns:
            Tuple of (parameters, observations) tensors
        """
        if prior_config is None:
            prior_config = {}
        
        if use_custom_prior:
            # Use the new CustomFlatPrior
            prior = self.create_custom_prior(prior_config=prior_config)
            theta = prior.sample((n_samples,))
            
            # Ensure center wavelengths are sorted for each sample
            if self.config.include_filter_params:
                centers = theta[..., prior.center_slice]
                sorted_centers, _ = torch.sort(centers, dim=-1)
                theta[..., prior.center_slice] = sorted_centers
            
        else:
            # Use the legacy method with individual priors
            priors = self.create_prior(prior_config)
            
            # Generate samples from priors
            theta = []
            
            # Concentration parameters
            concentration_prior_type = prior_config.get('concentration_prior_type', 'dirichlet')
            
            if concentration_prior_type == 'dirichlet':
                concentration = prior_config.get('concentration_params', {}).get('concentration', 1.0)
                prior = self.create_dirichlet_prior(concentration)
                concentration_samples = prior.sample((n_samples,))
                theta.append(concentration_samples)
                
            elif concentration_prior_type == 'beta':
                alpha = prior_config.get('concentration_params', {}).get('alpha', 1.0)
                beta = prior_config.get('concentration_params', {}).get('beta', 1.0)
                beta_prior = self.create_beta_prior(alpha, beta)
                concentration_samples = torch.stack([
                    beta_prior.sample((n_samples,)) for _ in range(self.n_fluorophores)
                ], dim=1)
                theta.append(concentration_samples)
            
            # Filter parameters
            if self.config.include_filter_params:
                # Center wavelengths
                center_low = prior_config.get('center_params', {}).get('low', self.config.center_wavelength_bounds[0])
                center_high = prior_config.get('center_params', {}).get('high', self.config.center_wavelength_bounds[1])
                center_prior = self.create_uniform_prior(center_low, center_high)
                center_samples = center_prior.sample((n_samples, self.config.n_channels))
                theta.append(center_samples)
                
                # Bandwidths
                bw_low = prior_config.get('bandwidth_params', {}).get('low', self.config.bandwidth_bounds[0])
                bw_high = prior_config.get('bandwidth_params', {}).get('high', self.config.bandwidth_bounds[1])
                bandwidth_prior = self.create_uniform_prior(bw_low, bw_high)
                bandwidth_samples = bandwidth_prior.sample((n_samples, self.config.n_channels))
                theta.append(bandwidth_samples)
            
            # Concatenate all parameters
            theta = torch.cat(theta, dim=1)
            
            # Ensure center wavelengths are sorted for each sample
            if self.config.include_filter_params:
                centers = theta[:, self.center_slice]
                sorted_centers, _ = torch.sort(centers, dim=1)
                theta[:, self.center_slice] = sorted_centers
        
        # Simulate observations
        if self.config.include_excitation_crosstalk and self.excitation_wavelengths:
            x = self.simulate_batch_with_excitation(theta.numpy(), add_noise)
        else:
            x = self.simulate_batch(theta.numpy(), add_noise)
        
        return theta, x
    
    def calculate_r_squared(
        self, 
        true_concentrations: np.ndarray, 
        predicted_concentrations: np.ndarray
    ) -> np.ndarray:
        """Calculate R² values for concentration predictions."""
        true_centered = true_concentrations - true_concentrations.mean(axis=1, keepdims=True)
        pred_centered = predicted_concentrations - predicted_concentrations.mean(axis=1, keepdims=True)
        
        numerator = np.sum(pred_centered * true_centered, axis=1) ** 2
        denominator = (np.sum(pred_centered**2, axis=1) * 
                      np.sum(true_centered**2, axis=1))
        
        r_squared = np.where(denominator > 0, numerator / denominator, 0.0)
        
        return r_squared
    
    def optimize_filter_configuration(
        self,
        n_trials: int = 100,
        prior_config: Dict = None
    ) -> Dict[str, List[float]]:
        """
        Optimize filter configuration using the enhanced simulator.
        
        Args:
            n_trials: Number of optimization trials
            prior_config: Configuration for prior distributions
            
        Returns:
            Dictionary with optimized center wavelengths and bandwidths
        """
        if prior_config is None:
            prior_config = {}
        
        # Generate training data with filter parameters
        theta, x = self.generate_training_data(
            n_samples=n_trials,
            prior_config=prior_config,
            add_noise=False
        )
        
        # Find the configuration with maximum signal-to-noise ratio
        signal_std = x.std(dim=0).mean(dim=0)
        signal_mean = x.mean(dim=0)
        
        # Calculate score for each configuration
        scores = []
        for i in range(n_trials):
            score = signal_mean[i].mean().item() / (signal_std[i].mean().item() + 1e-6)
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_theta = theta[best_idx]
        
        # Extract filter parameters from best configuration
        params = self.extract_parameters(best_theta.numpy())
        
        return {
            'center_wavelengths': params['center_wavelengths'][0].tolist(),
            'bandwidths': params['bandwidths'][0].tolist(),
            'score': float(np.max(scores))
        }


def create_enhanced_sbi_simulator(
    fluorophore_names: List[str],
    spectra_dir: Union[str, Path] = "data/spectra_npz",
    config: Optional[EnhancedSBIConfig] = None
) -> EnhancedSBISimulator:
    """
    Convenience function to create an enhanced SBI simulator.
    
    Args:
        fluorophore_names: List of fluorophore names
        spectra_dir: Directory containing spectra files
        config: Enhanced SBI configuration
        
    Returns:
        Configured enhanced SBI simulator
    """
    return EnhancedSBISimulator(fluorophore_names, spectra_dir, config)
