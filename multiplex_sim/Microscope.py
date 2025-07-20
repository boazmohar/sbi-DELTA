"""
Microscope simulation module for multiplexed fluorescence imaging.

This module provides functions for simulating fluorescence emission, detection signals,
and optimizing excitation wavelengths for multiplexed microscopy experiments.
"""

import numpy as np
import torch
from pathlib import Path
from scipy.optimize import differential_evolution
from typing import List, Dict, Tuple, Optional, Union
import warnings


class MicroscopeConfig:
    """Configuration class for microscope simulation parameters."""
    
    def __init__(
        self,
        wavelength_range: Tuple[int, int] = (400, 700),
        wavelength_step: float = 1.0,
        default_bandwidth: float = 30.0,
        default_num_channels: int = 2,
        optimization_seed: int = 42
    ):
        self.wavelength_range = wavelength_range
        self.wavelength_step = wavelength_step
        self.default_bandwidth = default_bandwidth
        self.default_num_channels = default_num_channels
        self.optimization_seed = optimization_seed
        
    @property
    def wavelengths(self) -> np.ndarray:
        """Generate wavelength array based on configuration."""
        return np.arange(self.wavelength_range[0], self.wavelength_range[1], self.wavelength_step)


def gaussian_emission(wavelengths: np.ndarray, λ_max: float, σ: float) -> np.ndarray:
    """
    Generate Gaussian emission spectrum.
    
    Args:
        wavelengths: Array of wavelength values (nm)
        λ_max: Peak wavelength (nm)
        σ: Standard deviation of the Gaussian (nm)
        
    Returns:
        Normalized Gaussian emission spectrum
    """
    if σ <= 0:
        raise ValueError("Standard deviation σ must be positive")
    
    return np.exp(-0.5 * ((wavelengths - λ_max) / σ)**2)


def create_channel_filters(
    wavelengths: np.ndarray,
    center_wavelengths: Optional[List[float]] = None,
    num_channels: int = 2,
    bandwidth: float = 30.0
) -> np.ndarray:
    """
    Create Gaussian filter responses for detection channels.
    
    Args:
        wavelengths: Array of wavelength values
        center_wavelengths: List of peak wavelengths for each channel
        num_channels: Number of detection channels (used if center_wavelengths is None)
        bandwidth: Standard deviation of each channel's filter response (nm)
        
    Returns:
        Array of shape (num_channels, len(wavelengths)) containing filter responses
    """
    if center_wavelengths is None:
        center_wavelengths = np.linspace(
            wavelengths[0] + 20, wavelengths[-1] - 20, num_channels
        )
    
    if len(center_wavelengths) != num_channels:
        raise ValueError(f"Number of center wavelengths ({len(center_wavelengths)}) "
                        f"must match num_channels ({num_channels})")
    
    channel_filters = []
    for cw in center_wavelengths:
        filter_response = np.exp(-0.5 * ((wavelengths - cw) / bandwidth)**2)
        channel_filters.append(filter_response)
    
    return np.stack(channel_filters)


def simulate_detected_signal(
    params: torch.Tensor,
    config: Optional[MicroscopeConfig] = None,
    num_channels: int = 2,
    center_wavelengths: Optional[List[float]] = None,
    bandwidth: float = 30.0
) -> torch.Tensor:
    """
    Simulate signal across multiple channels with narrow-band Gaussian filters.

    Args:
        params: Tensor of shape (batch_size, 2), with [λ_max, σ]
        config: MicroscopeConfig object (if None, uses default parameters)
        num_channels: Number of detection channels
        center_wavelengths: List of peak wavelengths for each channel
        bandwidth: Standard deviation of each channel's filter response (nm)

    Returns:
        Tensor of shape (batch_size, num_channels) containing simulated signals
    """
    if config is None:
        config = MicroscopeConfig()
    
    wavelengths = config.wavelengths
    
    # Create channel filters
    channel_filters = create_channel_filters(
        wavelengths, center_wavelengths, num_channels, bandwidth
    )
    
    signals = []
    for p in params:
        λ_max, σ = p.numpy()
        
        # Generate emission spectrum
        emission = gaussian_emission(wavelengths, λ_max, σ)
        
        # Calculate signal in each channel
        signal_vector = np.sum(channel_filters * emission, axis=1)
        signals.append(signal_vector)

    return torch.tensor(signals, dtype=torch.float32)


def simulate_photon_detection(
    intensities: torch.Tensor,
    photon_budget: float = 1000.0,
    background_rate: float = 10.0,
    add_noise: bool = True
) -> torch.Tensor:
    """
    Simulate photon detection with Poisson noise and background.
    
    Args:
        intensities: Tensor of shape (batch_size, num_channels) with relative intensities
        photon_budget: Total photon budget per measurement
        background_rate: Background photons per channel
        add_noise: Whether to add Poisson noise
        
    Returns:
        Tensor of detected photon counts with same shape as input
    """
    # Normalize intensities to sum to photon_budget
    normalized_intensities = intensities / intensities.sum(dim=1, keepdim=True) * photon_budget
    
    # Add background
    signal_with_background = normalized_intensities + background_rate
    
    if add_noise:
        # Apply Poisson noise
        return torch.poisson(signal_with_background)
    else:
        return signal_with_background


def load_excitation_spectra(
    fluor_names: List[str], 
    spectra_dir: Union[str, Path]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load excitation spectra from NPZ files.
    
    Args:
        fluor_names: List of fluorophore names
        spectra_dir: Directory containing NPZ files
        
    Returns:
        Dictionary mapping fluorophore names to (wavelengths, excitation) tuples
    """
    spectra_dir = Path(spectra_dir)
    if not spectra_dir.exists():
        raise FileNotFoundError(f"Spectra directory not found: {spectra_dir}")
    
    spectra = {}
    missing_files = []
    
    for name in fluor_names:
        path = spectra_dir / f"{name}.npz"
        if path.exists():
            try:
                data = np.load(path)
                wl = data["wavelengths_excitation"]
                excitation = data["excitation"]
                
                # Normalize excitation spectrum
                if excitation.max() > 0:
                    excitation = excitation / excitation.max()
                else:
                    warnings.warn(f"Zero excitation spectrum for {name}")
                
                spectra[name] = (wl, excitation)
            except Exception as e:
                warnings.warn(f"Error loading {name}.npz: {e}")
                missing_files.append(name)
        else:
            missing_files.append(name)
    
    if missing_files:
        warnings.warn(f"Missing spectra files for: {missing_files}")
    
    return spectra


def calculate_crosstalk_matrix(
    excitation_wavelengths: List[float],
    spectra_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Calculate crosstalk matrix for given excitation wavelengths.
    
    Args:
        excitation_wavelengths: List of excitation wavelengths for each fluorophore
        spectra_dict: Dictionary of fluorophore spectra
        
    Returns:
        Crosstalk matrix where element (i,j) is excitation of fluorophore j by laser i
    """
    n_fluors = len(excitation_wavelengths)
    crosstalk_matrix = np.zeros((n_fluors, n_fluors))
    
    fluor_names = list(spectra_dict.keys())
    shared_wl = list(spectra_dict.values())[0][0]
    
    for i, λi in enumerate(excitation_wavelengths):
        for j, name_j in enumerate(fluor_names):
            _, ex_j = spectra_dict[name_j]
            idx = np.clip(np.searchsorted(shared_wl, λi, side="left"), 0, len(ex_j) - 1)
            crosstalk_matrix[i, j] = ex_j[idx]
    
    return crosstalk_matrix


def excitation_cost(
    wavelengths: np.ndarray,
    spectra_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    shared_wl: np.ndarray,
    verbose: bool = False
) -> float:
    """
    Calculate cost function for excitation wavelength optimization.
    
    Args:
        wavelengths: Array of excitation wavelengths to evaluate
        spectra_dict: Dictionary of fluorophore spectra
        shared_wl: Shared wavelength grid
        verbose: Whether to print detailed cost breakdown
        
    Returns:
        Total cost (lower is better)
    """
    total_cost = 0.0
    fluor_names = list(spectra_dict.keys())
    
    for i, (name_i, λi) in enumerate(zip(fluor_names, wavelengths)):
        _, ex_i = spectra_dict[name_i]
        idx_i = np.clip(np.searchsorted(shared_wl, λi, side="left"), 0, len(ex_i) - 1)
        self_signal = ex_i[idx_i]

        crosstalk = 0.0
        for j, name_j in enumerate(fluor_names):
            if i == j:
                continue
            _, ex_j = spectra_dict[name_j]
            idx_j = np.clip(np.searchsorted(shared_wl, λi, side="left"), 0, len(ex_j) - 1)
            crosstalk += ex_j[idx_j]

        if verbose:
            print(f"{name_i} @ {λi:.1f} nm → self: {self_signal:.3f}, "
                  f"crosstalk: {crosstalk:.3f}, net: {-self_signal + crosstalk:.3f}")

        # Cost function: maximize self-signal, minimize crosstalk
        total_cost += -self_signal + crosstalk

    return total_cost


def find_optimal_excitation(
    fluor_names: List[str],
    spectra_dir: Union[str, Path],
    search_range: float = 30.0,
    config: Optional[MicroscopeConfig] = None
) -> Dict[str, int]:
    """
    Find optimal excitation wavelengths for a set of fluorophores.
    
    Args:
        fluor_names: List of fluorophore names
        spectra_dir: Directory containing spectra files
        search_range: Search range around peak excitation (nm)
        config: MicroscopeConfig object
        
    Returns:
        Dictionary mapping fluorophore names to optimal excitation wavelengths
    """
    if config is None:
        config = MicroscopeConfig()
    
    # Load spectra
    spectra = load_excitation_spectra(fluor_names, spectra_dir)
    if len(spectra) != len(fluor_names):
        missing = set(fluor_names) - set(spectra.keys())
        raise ValueError(f"Missing spectra for: {missing}")

    # Get shared wavelength grid
    shared_wavelength = np.asarray(list(spectra.values())[0][0])
    
    # Define search bounds around peak excitation for each fluorophore
    bounds = []
    for name in fluor_names:
        wl, ex = spectra[name]
        max_idx = np.argmax(ex)
        peak_wl = wl[max_idx]
        lower = max(wl[0], peak_wl - search_range)
        upper = min(wl[-1], peak_wl + search_range)
        bounds.append((lower, upper))

    # Optimize excitation wavelengths
    result = differential_evolution(
        func=excitation_cost,
        bounds=bounds,
        args=(spectra, shared_wavelength, False),  # Set verbose=False for optimization
        strategy='best1bin',
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        seed=config.optimization_seed
    )

    optimal_wavelengths = np.round(result.x).astype(int)
    
    # Print final cost breakdown
    print("Optimal excitation wavelengths found:")
    excitation_cost(result.x, spectra, shared_wavelength, verbose=True)
    
    return dict(zip(fluor_names, optimal_wavelengths))


def generate_dye_combinations(
    dye_names: List[str],
    n_combinations: int = 1000,
    concentration_range: Tuple[float, float] = (0.0, 1.0),
    min_active_dyes: int = 1
) -> torch.Tensor:
    """
    Generate random dye concentration combinations for simulation.
    
    Args:
        dye_names: List of dye names
        n_combinations: Number of combinations to generate
        concentration_range: Range of concentrations (min, max)
        min_active_dyes: Minimum number of dyes with non-zero concentration
        
    Returns:
        Tensor of shape (n_combinations, len(dye_names)) with concentrations
    """
    n_dyes = len(dye_names)
    combinations = []
    
    for _ in range(n_combinations):
        # Generate random concentrations
        concentrations = torch.rand(n_dyes) * (concentration_range[1] - concentration_range[0]) + concentration_range[0]
        
        # Ensure minimum number of active dyes
        if min_active_dyes > 0:
            active_indices = torch.randperm(n_dyes)[:min_active_dyes]
            mask = torch.zeros(n_dyes, dtype=torch.bool)
            mask[active_indices] = True
            
            # Set inactive dyes to zero with some probability
            inactive_mask = ~mask
            zero_prob = torch.rand(inactive_mask.sum()) < 0.3  # 30% chance of being zero
            concentrations[inactive_mask] *= ~zero_prob
        
        combinations.append(concentrations)
    
    return torch.stack(combinations)
