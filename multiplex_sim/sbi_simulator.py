"""
SBI Simulator module for multiplexed fluorescence microscopy.

This module provides a unified interface for simulation-based inference (SBI) 
of fluorophore concentrations from multiplexed microscopy data.
"""

import numpy as np
import torch
from torch.distributions import Dirichlet, Beta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
from scipy.interpolate import interp1d
import warnings
from dataclasses import dataclass

from .Microscope import find_optimal_excitation
from .io import list_fluorophores


@dataclass
class SBIConfig:
    """Configuration for SBI simulation parameters."""
    
    # Spectral parameters
    wavelength_range: Tuple[int, int] = (500, 800)
    wavelength_step: float = 1.0
    
    # Photon budget parameters
    total_dye_photons: float = 300.0
    total_background_photons: float = 30.0
    
    # Filter parameters
    filter_type: str = "sigmoid"  # "sigmoid" or "gaussian"
    edge_steepness: float = 1.0
    
    # Background fluorophore
    background_fluorophore: str = "NADH"
    
    # Excitation parameters
    excitation_wavelengths: Optional[List[float]] = None  # Manual specification
    optimize_excitation: bool = True  # Auto-optimize if wavelengths not provided
    excitation_search_range: float = 30.0  # Search range around peak excitation
    include_excitation_crosstalk: bool = True  # Model crosstalk effects
    excitation_power_budget: float = 1.0  # Total excitation power (normalized)
    
    # Random seed
    random_seed: int = 42


class FilterBank:
    """Class for managing detection filter configurations."""
    
    def __init__(self, config: SBIConfig):
        self.config = config
        self.wavelengths = np.arange(
            config.wavelength_range[0], 
            config.wavelength_range[1] + config.wavelength_step, 
            config.wavelength_step
        )
    
    def create_sigmoid_filters(
        self, 
        center_wavelengths: List[float], 
        bandwidths: List[float]
    ) -> np.ndarray:
        """
        Create sigmoid-edge bandpass filters.
        
        Args:
            center_wavelengths: Center wavelengths for each filter
            bandwidths: Bandwidth (FWHM) for each filter
            
        Returns:
            Array of shape (n_filters, n_wavelengths)
        """
        filters = []
        for center, bandwidth in zip(center_wavelengths, bandwidths):
            # Sigmoid edges for bandpass filter
            left_edge = center - bandwidth / 2
            right_edge = center + bandwidth / 2
            
            left_sigmoid = 1 / (1 + np.exp(-(self.wavelengths - left_edge) * self.config.edge_steepness))
            right_sigmoid = 1 / (1 + np.exp((self.wavelengths - right_edge) * self.config.edge_steepness))
            
            bandpass = left_sigmoid * right_sigmoid
            filters.append(bandpass)
        
        return np.stack(filters)
    
    def create_gaussian_filters(
        self, 
        center_wavelengths: List[float], 
        bandwidths: List[float]
    ) -> np.ndarray:
        """
        Create Gaussian bandpass filters.
        
        Args:
            center_wavelengths: Center wavelengths for each filter
            bandwidths: Standard deviation for each filter
            
        Returns:
            Array of shape (n_filters, n_wavelengths)
        """
        filters = []
        for center, sigma in zip(center_wavelengths, bandwidths):
            gaussian = np.exp(-0.5 * ((self.wavelengths - center) / sigma)**2)
            filters.append(gaussian)
        
        return np.stack(filters)
    
    def create_filters(
        self, 
        center_wavelengths: List[float], 
        bandwidths: List[float]
    ) -> np.ndarray:
        """Create filters based on configuration."""
        if self.config.filter_type == "sigmoid":
            return self.create_sigmoid_filters(center_wavelengths, bandwidths)
        elif self.config.filter_type == "gaussian":
            return self.create_gaussian_filters(center_wavelengths, bandwidths)
        else:
            raise ValueError(f"Unknown filter type: {self.config.filter_type}")


class SpectraManager:
    """Class for managing fluorophore spectra data."""
    
    def __init__(self, spectra_dir: Union[str, Path], config: SBIConfig):
        self.spectra_dir = Path(spectra_dir)
        self.config = config
        self.wavelengths = np.arange(
            config.wavelength_range[0], 
            config.wavelength_range[1] + config.wavelength_step, 
            config.wavelength_step
        )
        self._interpolated_spectra = {}
        self._background_spectrum = None
    
    def load_and_interpolate_emission(self, fluorophore_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Load and interpolate emission spectra onto common wavelength grid.
        
        Args:
            fluorophore_names: List of fluorophore names
            
        Returns:
            Dictionary mapping fluorophore names to interpolated emission spectra
        """
        spectra = {}
        
        for name in fluorophore_names:
            path = self.spectra_dir / f"{name}.npz"
            if not path.exists():
                warnings.warn(f"Spectra file not found: {path}")
                continue
            
            try:
                data = np.load(path)
                wl = data["wavelengths_emission"]
                em = data["emission"]
                
                # Normalize to peak
                if em.max() > 0:
                    em = em / em.max()
                
                # Interpolate onto common grid
                interp_func = interp1d(wl, em, bounds_error=False, fill_value=0.0)
                interpolated = interp_func(self.wavelengths)
                
                spectra[name] = interpolated
                
            except Exception as e:
                warnings.warn(f"Error loading {name}: {e}")
        
        self._interpolated_spectra.update(spectra)
        return spectra
    
    def load_and_interpolate_excitation(self, fluorophore_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Load and interpolate excitation spectra onto common wavelength grid.
        
        Args:
            fluorophore_names: List of fluorophore names
            
        Returns:
            Dictionary mapping fluorophore names to interpolated excitation spectra
        """
        spectra = {}
        
        for name in fluorophore_names:
            path = self.spectra_dir / f"{name}.npz"
            if not path.exists():
                warnings.warn(f"Excitation spectra file not found: {path}")
                continue
            
            try:
                data = np.load(path)
                wl = data["wavelengths_excitation"]
                ex = data["excitation"]
                
                # Normalize to peak
                if ex.max() > 0:
                    ex = ex / ex.max()
                
                # Interpolate onto common grid
                interp_func = interp1d(wl, ex, bounds_error=False, fill_value=0.0)
                interpolated = interp_func(self.wavelengths)
                
                spectra[name] = interpolated
                
            except Exception as e:
                warnings.warn(f"Error loading excitation for {name}: {e}")
        
        return spectra
    
    def load_background_spectrum(self, background_name: str = "NADH") -> np.ndarray:
        """Load and interpolate background fluorescence spectrum."""
        if self._background_spectrum is not None:
            return self._background_spectrum
        
        path = self.spectra_dir / f"{background_name}.npz"
        if not path.exists():
            warnings.warn(f"Background spectrum not found: {path}")
            return np.zeros_like(self.wavelengths)
        
        try:
            data = np.load(path)
            wl = data["wavelengths_emission"]
            em = data["emission"]
            
            # Normalize
            if em.max() > 0:
                em = em / em.max()
            
            # Interpolate
            interp_func = interp1d(wl, em, bounds_error=False, fill_value=0.0)
            self._background_spectrum = interp_func(self.wavelengths)
            
        except Exception as e:
            warnings.warn(f"Error loading background spectrum: {e}")
            self._background_spectrum = np.zeros_like(self.wavelengths)
        
        return self._background_spectrum
    
    def get_interpolated_spectra(self) -> Dict[str, np.ndarray]:
        """Get cached interpolated spectra."""
        return self._interpolated_spectra.copy()


class SBISimulator:
    """
    Main simulator class for SBI-based fluorophore concentration inference.
    """
    
    def __init__(
        self, 
        fluorophore_names: List[str],
        spectra_dir: Union[str, Path],
        config: Optional[SBIConfig] = None
    ):
        self.fluorophore_names = fluorophore_names
        self.config = config or SBIConfig()
        
        # Initialize components
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
        
        # Set up excitation wavelengths
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
    
    def _optimize_excitation_wavelengths(self) -> List[float]:
        """
        Optimize excitation wavelengths for the loaded fluorophores.
        
        Returns:
            List of optimal excitation wavelengths
        """
        try:
            # Try advanced optimization first
            from .advanced_optimization import find_optimal_excitation_advanced, OptimizationConfig
            
            # Create optimization config
            opt_config = OptimizationConfig(
                search_range=self.config.excitation_search_range,
                min_wavelength_separation=10.0,
                n_multistart=5,  # Reduced for faster execution
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
        """
        Get peak excitation wavelengths for each fluorophore.
        
        Returns:
            List of peak excitation wavelengths
        """
        peak_wavelengths = []
        for name in self.fluorophore_names:
            if name in self.excitation_spectra:
                spectrum = self.excitation_spectra[name]
                peak_idx = np.argmax(spectrum)
                peak_wl = self.spectra_manager.wavelengths[peak_idx]
                peak_wavelengths.append(peak_wl)
            else:
                # Default fallback
                peak_wavelengths.append(500.0)
                warnings.warn(f"No excitation spectrum for {name}, using default 500nm")
        
        return peak_wavelengths
    
    def calculate_excitation_crosstalk_matrix(
        self, 
        excitation_wavelengths: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Calculate excitation crosstalk matrix.
        
        Args:
            excitation_wavelengths: List of excitation wavelengths (one per fluorophore)
            
        Returns:
            Crosstalk matrix of shape (n_fluorophores, n_fluorophores)
            where element (i,j) is the excitation of fluorophore j by laser i
        """
        if excitation_wavelengths is None:
            excitation_wavelengths = self.excitation_wavelengths
        
        if excitation_wavelengths is None:
            # No excitation modeling, return identity matrix
            n_fluors = len(self.fluorophore_names)
            return np.eye(n_fluors)
        
        n_fluors = len(self.fluorophore_names)
        crosstalk_matrix = np.zeros((n_fluors, n_fluors))
        
        for i, exc_wl in enumerate(excitation_wavelengths):
            for j, fluor_name in enumerate(self.fluorophore_names):
                if fluor_name in self.excitation_spectra:
                    # Find closest wavelength index
                    wl_idx = np.argmin(np.abs(self.spectra_manager.wavelengths - exc_wl))
                    crosstalk_matrix[i, j] = self.excitation_spectra[fluor_name][wl_idx]
                else:
                    # No excitation spectrum available
                    crosstalk_matrix[i, j] = 1.0 if i == j else 0.0
        
        return crosstalk_matrix
    
    def calculate_background_excitation_response(
        self,
        excitation_wavelengths: Optional[List[float]] = None,
        laser_powers: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate background fluorescence response to excitation lasers.
        
        Args:
            excitation_wavelengths: List of excitation wavelengths
            laser_powers: Array of laser powers (if None, assumes equal power)
            
        Returns:
            Total background excitation response
        """
        if self.background_excitation is None or excitation_wavelengths is None:
            return 1.0  # Default constant background
        
        if laser_powers is None:
            laser_powers = np.ones(len(excitation_wavelengths))
        
        total_bg_excitation = 0.0
        for exc_wl, power in zip(excitation_wavelengths, laser_powers):
            # Find closest wavelength index
            wl_idx = np.argmin(np.abs(self.spectra_manager.wavelengths - exc_wl))
            bg_response = self.background_excitation[wl_idx]
            total_bg_excitation += power * bg_response
        
        return total_bg_excitation
    
    def simulate_batch_with_excitation(
        self,
        concentrations: np.ndarray,
        center_wavelengths: List[float],
        bandwidths: List[float],
        excitation_wavelengths: Optional[List[float]] = None,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulate detected photon counts with excitation crosstalk modeling.
        
        Args:
            concentrations: Array of shape (batch_size, n_fluorophores)
            center_wavelengths: Center wavelengths for detection filters
            bandwidths: Bandwidths for detection filters
            excitation_wavelengths: Excitation wavelengths (one per fluorophore)
            add_noise: Whether to add Poisson noise
            
        Returns:
            Tensor of shape (batch_size, n_channels) with detected photon counts
        """
        if not self.config.include_excitation_crosstalk:
            # Fall back to simple simulation
            return self.simulate_batch(concentrations, center_wavelengths, bandwidths, add_noise)
        
        if excitation_wavelengths is None:
            excitation_wavelengths = self.excitation_wavelengths
        
        if excitation_wavelengths is None:
            warnings.warn("No excitation wavelengths available, using simple simulation")
            return self.simulate_batch(concentrations, center_wavelengths, bandwidths, add_noise)
        
        batch_size, n_fluorophores = concentrations.shape
        n_channels = len(center_wavelengths)
        
        # Create detection filters
        filters = self.filter_bank.create_filters(center_wavelengths, bandwidths)
        
        # Calculate excitation crosstalk matrix
        crosstalk_matrix = self.calculate_excitation_crosstalk_matrix(excitation_wavelengths)
        
        # Prepare emission spectra matrix
        emission_matrix = np.stack([
            self.emission_spectra[name] for name in self.fluorophore_names
        ])  # Shape: (n_fluorophores, n_wavelengths)
        
        results = []
        
        for batch_idx in range(batch_size):
            conc = concentrations[batch_idx]  # Shape: (n_fluorophores,)
            
            # Apply excitation crosstalk: each laser excites multiple fluorophores
            excited_amplitudes = np.zeros(n_fluorophores)
            for i, laser_power in enumerate(conc):  # Assume laser power proportional to concentration
                if laser_power > 0:
                    # This laser excites all fluorophores according to crosstalk matrix
                    excited_amplitudes += laser_power * crosstalk_matrix[i, :]
            
            # Calculate dye signals for each channel
            dye_signals = np.zeros(n_channels)
            for i, excited_amp in enumerate(excited_amplitudes):
                if excited_amp > 0:
                    spectrum = emission_matrix[i]  # Shape: (n_wavelengths,)
                    # Integrate spectrum through each filter
                    channel_responses = np.sum(filters * spectrum, axis=1)  # Shape: (n_channels,)
                    dye_signals += excited_amp * channel_responses
            
            # Normalize dye signals to photon budget
            total_dye_signal = dye_signals.sum()
            if total_dye_signal > 0:
                dye_photons = dye_signals / total_dye_signal * self.config.total_dye_photons
            else:
                dye_photons = np.zeros(n_channels)
            
            # Add background signal with excitation-dependent response
            bg_excitation_response = self.calculate_background_excitation_response(
                excitation_wavelengths, conc
            )
            background_responses = np.sum(filters * self.background_spectrum, axis=1)
            background_total = background_responses.sum()
            if background_total > 0:
                background_photons = (background_responses / background_total * 
                                    self.config.total_background_photons * bg_excitation_response)
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
    
    def simulate_batch(
        self,
        concentrations: np.ndarray,
        center_wavelengths: List[float],
        bandwidths: List[float],
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulate detected photon counts for a batch of concentration combinations.
        
        Args:
            concentrations: Array of shape (batch_size, n_fluorophores)
            center_wavelengths: Center wavelengths for detection filters
            bandwidths: Bandwidths for detection filters
            add_noise: Whether to add Poisson noise
            
        Returns:
            Tensor of shape (batch_size, n_channels) with detected photon counts
        """
        batch_size, n_fluorophores = concentrations.shape
        n_channels = len(center_wavelengths)
        
        # Create detection filters
        filters = self.filter_bank.create_filters(center_wavelengths, bandwidths)
        
        # Prepare emission spectra matrix
        emission_matrix = np.stack([
            self.emission_spectra[name] for name in self.fluorophore_names
        ])  # Shape: (n_fluorophores, n_wavelengths)
        
        results = []
        
        for batch_idx in range(batch_size):
            conc = concentrations[batch_idx]  # Shape: (n_fluorophores,)
            
            # Calculate dye signals for each channel
            dye_signals = np.zeros(n_channels)
            for i, fluorophore_conc in enumerate(conc):
                if fluorophore_conc > 0:
                    spectrum = emission_matrix[i]  # Shape: (n_wavelengths,)
                    # Integrate spectrum through each filter
                    channel_responses = np.sum(filters * spectrum, axis=1)  # Shape: (n_channels,)
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
                background_photons = (background_responses / background_total * 
                                    self.config.total_background_photons)
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
        """
        Create Dirichlet prior for concentration ratios.
        
        Args:
            concentration: Concentration parameter for Dirichlet distribution
            
        Returns:
            Dirichlet distribution over concentration ratios
        """
        return Dirichlet(concentration * torch.ones(len(self.fluorophore_names)))
    
    def create_beta_prior(self, alpha: float = 1.0, beta: float = 1.0) -> Beta:
        """
        Create Beta prior for individual concentrations.
        
        Args:
            alpha: Alpha parameter for Beta distribution
            beta: Beta parameter for Beta distribution
            
        Returns:
            Beta distribution for individual concentrations
        """
        return Beta(alpha, beta)
    
    def generate_training_data(
        self,
        n_samples: int,
        center_wavelengths: List[float],
        bandwidths: List[float],
        prior_type: str = "dirichlet",
        prior_params: Dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for SBI.
        
        Args:
            n_samples: Number of training samples
            center_wavelengths: Center wavelengths for detection filters
            bandwidths: Bandwidths for detection filters
            prior_type: Type of prior ("dirichlet" or "beta")
            prior_params: Parameters for the prior distribution
            
        Returns:
            Tuple of (parameters, observations) tensors
        """
        if prior_params is None:
            prior_params = {}
        
        # Generate parameters from prior
        if prior_type == "dirichlet":
            concentration = prior_params.get("concentration", 1.0)
            prior = self.create_dirichlet_prior(concentration)
            theta = prior.sample((n_samples,))
        elif prior_type == "beta":
            alpha = prior_params.get("alpha", 1.0)
            beta = prior_params.get("beta", 1.0)
            prior = self.create_beta_prior(alpha, beta)
            # Sample independent Beta for each fluorophore
            theta = torch.stack([
                prior.sample((n_samples,)) for _ in range(len(self.fluorophore_names))
            ], dim=1)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Simulate observations (use excitation-aware simulation if configured)
        if self.config.include_excitation_crosstalk and self.excitation_wavelengths:
            x = self.simulate_batch_with_excitation(
                theta.numpy(), 
                center_wavelengths, 
                bandwidths, 
                add_noise=True
            )
        else:
            x = self.simulate_batch(
                theta.numpy(), 
                center_wavelengths, 
                bandwidths, 
                add_noise=True
            )
        
        return theta, x
    
    def calculate_r_squared(
        self, 
        true_concentrations: np.ndarray, 
        predicted_concentrations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate R² values for concentration predictions.
        
        Args:
            true_concentrations: True concentration values
            predicted_concentrations: Predicted concentration values
            
        Returns:
            Array of R² values
        """
        # Center the data
        true_centered = true_concentrations - true_concentrations.mean(axis=1, keepdims=True)
        pred_centered = predicted_concentrations - predicted_concentrations.mean(axis=1, keepdims=True)
        
        # Calculate R²
        numerator = np.sum(pred_centered * true_centered, axis=1) ** 2
        denominator = (np.sum(pred_centered**2, axis=1) * 
                      np.sum(true_centered**2, axis=1))
        
        # Avoid division by zero
        r_squared = np.where(denominator > 0, numerator / denominator, 0.0)
        
        return r_squared
    
    def optimize_filter_configuration(
        self,
        n_channels: int,
        wavelength_bounds: Tuple[float, float] = None,
        bandwidth_bounds: Tuple[float, float] = (10.0, 50.0),
        n_trials: int = 100
    ) -> Dict[str, List[float]]:
        """
        Optimize filter configuration for maximum multiplexing performance.
        
        Args:
            n_channels: Number of detection channels
            wavelength_bounds: Bounds for center wavelengths
            bandwidth_bounds: Bounds for filter bandwidths
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimized center wavelengths and bandwidths
        """
        if wavelength_bounds is None:
            wavelength_bounds = (self.config.wavelength_range[0] + 50, 
                               self.config.wavelength_range[1] - 50)
        
        best_score = -np.inf
        best_config = None
        
        for trial in range(n_trials):
            # Random configuration
            centers = np.sort(self.rng.uniform(
                wavelength_bounds[0], wavelength_bounds[1], n_channels
            ))
            bandwidths = self.rng.uniform(
                bandwidth_bounds[0], bandwidth_bounds[1], n_channels
            )
            
            # Evaluate configuration
            try:
                # Generate test data
                theta, x = self.generate_training_data(
                    n_samples=100,
                    center_wavelengths=centers.tolist(),
                    bandwidths=bandwidths.tolist(),
                    prior_type="dirichlet"
                )
                
                # Simple score: signal separation
                signal_std = x.std(dim=0).mean().item()
                signal_mean = x.mean(dim=0).mean().item()
                score = signal_mean / (signal_std + 1e-6)  # Signal-to-noise-like metric
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        "center_wavelengths": centers.tolist(),
                        "bandwidths": bandwidths.tolist(),
                        "score": score
                    }
                    
            except Exception as e:
                warnings.warn(f"Trial {trial} failed: {e}")
                continue
        
        if best_config is None:
            raise RuntimeError("Filter optimization failed")
        
        return best_config


def create_sbi_simulator(
    fluorophore_names: List[str],
    spectra_dir: Union[str, Path] = "data/spectra_npz",
    config: Optional[SBIConfig] = None
) -> SBISimulator:
    """
    Convenience function to create an SBI simulator.
    
    Args:
        fluorophore_names: List of fluorophore names
        spectra_dir: Directory containing spectra files
        config: SBI configuration
        
    Returns:
        Configured SBI simulator
    """
    return SBISimulator(fluorophore_names, spectra_dir, config)
