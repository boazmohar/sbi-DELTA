"""
Advanced excitation wavelength optimization for multiplexed microscopy.

This module provides improved algorithms for optimizing excitation wavelengths
that consider signal quality, crosstalk, detection channel separability, and
background interference in a multi-objective framework.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import time

from .io import list_fluorophores


@dataclass
class OptimizationConfig:
    """Configuration for advanced excitation optimization."""
    
    # Search parameters
    search_range: float = 50.0  # Search range around peak excitation (nm)
    min_wavelength_separation: float = 10.0  # Minimum separation between lasers (nm)
    
    # Cost function weights
    signal_strength_weight: float = 1.0
    crosstalk_weight: float = 2.0
    detection_separability_weight: float = 1.5
    background_weight: float = 0.5
    separation_penalty_weight: float = 3.0
    
    # Optimization parameters
    n_multistart: int = 10  # Number of multi-start runs
    population_size: int = 20  # Population size for differential evolution
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Detection configuration
    detection_centers: Optional[List[float]] = None
    detection_bandwidths: Optional[List[float]] = None
    photon_budget: float = 300.0
    background_photons: float = 30.0
    
    # Validation parameters
    n_validation_samples: int = 1000
    validation_noise_level: float = 0.1
    
    # Random seed
    random_seed: int = 42


class AdvancedExcitationOptimizer:
    """Advanced excitation wavelength optimizer with multi-objective cost function."""
    
    def __init__(
        self,
        fluorophore_names: List[str],
        spectra_dir: Union[str, Path],
        config: Optional[OptimizationConfig] = None
    ):
        self.fluorophore_names = fluorophore_names
        self.spectra_dir = Path(spectra_dir)
        self.config = config or OptimizationConfig()
        
        # Load spectra
        self.excitation_spectra = self._load_excitation_spectra()
        self.emission_spectra = self._load_emission_spectra()
        self.background_excitation = self._load_background_excitation()
        
        # Set up wavelength grid
        self.wavelength_grid = self._create_wavelength_grid()
        
        # Set up random number generator
        np.random.seed(self.config.random_seed)
        
        # Cache for expensive calculations
        self._cache = {}
    
    def _load_excitation_spectra(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load and validate excitation spectra."""
        spectra = {}
        
        for name in self.fluorophore_names:
            path = self.spectra_dir / f"{name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Spectra file not found: {path}")
            
            try:
                data = np.load(path)
                wl = data["wavelengths_excitation"]
                ex = data["excitation"]
                
                # Normalize and validate
                if ex.max() <= 0:
                    raise ValueError(f"Invalid excitation spectrum for {name}")
                
                ex_normalized = ex / ex.max()
                spectra[name] = (wl, ex_normalized)
                
            except Exception as e:
                raise RuntimeError(f"Error loading excitation spectrum for {name}: {e}")
        
        return spectra
    
    def _load_emission_spectra(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load and validate emission spectra."""
        spectra = {}
        
        for name in self.fluorophore_names:
            path = self.spectra_dir / f"{name}.npz"
            if not path.exists():
                continue
            
            try:
                data = np.load(path)
                wl = data["wavelengths_emission"]
                em = data["emission"]
                
                if em.max() > 0:
                    em_normalized = em / em.max()
                    spectra[name] = (wl, em_normalized)
                
            except Exception as e:
                warnings.warn(f"Error loading emission spectrum for {name}: {e}")
        
        return spectra
    
    def _load_background_excitation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load background excitation spectrum (e.g., NADH)."""
        bg_names = ["NADH", "NADH - ", "nadh"]
        
        for bg_name in bg_names:
            path = self.spectra_dir / f"{bg_name}.npz"
            if path.exists():
                try:
                    data = np.load(path)
                    wl = data["wavelengths_excitation"]
                    ex = data["excitation"]
                    
                    if ex.max() > 0:
                        ex_normalized = ex / ex.max()
                        return (wl, ex_normalized)
                
                except Exception as e:
                    warnings.warn(f"Error loading background excitation: {e}")
        
        warnings.warn("Background excitation spectrum not found")
        return None
    
    def _create_wavelength_grid(self) -> np.ndarray:
        """Create common wavelength grid for interpolation."""
        all_wavelengths = []
        
        # Collect all wavelength ranges
        for wl, _ in self.excitation_spectra.values():
            all_wavelengths.extend([wl.min(), wl.max()])
        
        if self.background_excitation:
            wl, _ = self.background_excitation
            all_wavelengths.extend([wl.min(), wl.max()])
        
        # Create grid covering all spectra
        wl_min = max(300, min(all_wavelengths))  # Reasonable lower bound
        wl_max = min(800, max(all_wavelengths))  # Reasonable upper bound
        
        return np.arange(wl_min, wl_max + 1, 1.0)
    
    def _interpolate_spectrum(
        self, 
        wavelengths: np.ndarray, 
        spectrum: np.ndarray
    ) -> np.ndarray:
        """Interpolate spectrum onto common wavelength grid."""
        interp_func = interp1d(
            wavelengths, spectrum, 
            bounds_error=False, fill_value=0.0, kind='linear'
        )
        return interp_func(self.wavelength_grid)
    
    def calculate_signal_strength(self, excitation_wavelengths: np.ndarray) -> float:
        """Calculate total signal strength for given excitation wavelengths."""
        total_signal = 0.0
        
        for i, (name, exc_wl) in enumerate(zip(self.fluorophore_names, excitation_wavelengths)):
            if name in self.excitation_spectra:
                wl, ex = self.excitation_spectra[name]
                
                # Interpolate excitation spectrum
                ex_interp = self._interpolate_spectrum(wl, ex)
                
                # Find excitation efficiency at chosen wavelength
                wl_idx = np.argmin(np.abs(self.wavelength_grid - exc_wl))
                signal_strength = ex_interp[wl_idx]
                
                total_signal += signal_strength
        
        return total_signal
    
    def calculate_crosstalk_penalty(self, excitation_wavelengths: np.ndarray) -> float:
        """Calculate crosstalk penalty for given excitation wavelengths."""
        total_crosstalk = 0.0
        n_fluors = len(self.fluorophore_names)
        
        for i, exc_wl_i in enumerate(excitation_wavelengths):
            for j, name_j in enumerate(self.fluorophore_names):
                if i == j:
                    continue  # Skip self-excitation
                
                if name_j in self.excitation_spectra:
                    wl, ex = self.excitation_spectra[name_j]
                    ex_interp = self._interpolate_spectrum(wl, ex)
                    
                    # Find crosstalk at laser i wavelength
                    wl_idx = np.argmin(np.abs(self.wavelength_grid - exc_wl_i))
                    crosstalk = ex_interp[wl_idx]
                    
                    total_crosstalk += crosstalk
        
        return total_crosstalk
    
    def calculate_detection_separability(
        self, 
        excitation_wavelengths: np.ndarray
    ) -> float:
        """Calculate detection channel separability."""
        if (self.config.detection_centers is None or 
            self.config.detection_bandwidths is None):
            return 0.0  # No detection info available
        
        # Simulate emission signals in detection channels
        n_channels = len(self.config.detection_centers)
        signal_matrix = np.zeros((len(self.fluorophore_names), n_channels))
        
        for i, (name, exc_wl) in enumerate(zip(self.fluorophore_names, excitation_wavelengths)):
            if name not in self.excitation_spectra or name not in self.emission_spectra:
                continue
            
            # Get excitation efficiency
            wl_ex, ex = self.excitation_spectra[name]
            ex_interp = self._interpolate_spectrum(wl_ex, ex)
            wl_idx = np.argmin(np.abs(self.wavelength_grid - exc_wl))
            excitation_eff = ex_interp[wl_idx]
            
            # Get emission spectrum
            wl_em, em = self.emission_spectra[name]
            em_interp = self._interpolate_spectrum(wl_em, em)
            
            # Calculate signal in each detection channel
            for j, (center, bandwidth) in enumerate(zip(
                self.config.detection_centers, self.config.detection_bandwidths
            )):
                # Gaussian detection filter
                detection_filter = np.exp(
                    -0.5 * ((self.wavelength_grid - center) / bandwidth)**2
                )
                
                # Integrate emission through filter
                channel_signal = np.sum(em_interp * detection_filter) * excitation_eff
                signal_matrix[i, j] = channel_signal
        
        # Calculate separability as condition number (lower is better)
        if signal_matrix.size > 0:
            try:
                # Use SVD for numerical stability
                U, s, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
                condition_number = s.max() / (s.min() + 1e-12)
                separability = 1.0 / (1.0 + condition_number)  # Convert to 0-1 scale
                return separability
            except:
                return 0.0
        
        return 0.0
    
    def calculate_background_penalty(self, excitation_wavelengths: np.ndarray) -> float:
        """Calculate background excitation penalty."""
        if self.background_excitation is None:
            return 0.0
        
        wl, bg_ex = self.background_excitation
        bg_ex_interp = self._interpolate_spectrum(wl, bg_ex)
        
        total_background = 0.0
        for exc_wl in excitation_wavelengths:
            wl_idx = np.argmin(np.abs(self.wavelength_grid - exc_wl))
            background_response = bg_ex_interp[wl_idx]
            total_background += background_response
        
        return total_background
    
    def calculate_separation_penalty(self, excitation_wavelengths: np.ndarray) -> float:
        """Calculate penalty for insufficient wavelength separation."""
        penalty = 0.0
        min_sep = self.config.min_wavelength_separation
        
        for i in range(len(excitation_wavelengths)):
            for j in range(i + 1, len(excitation_wavelengths)):
                separation = abs(excitation_wavelengths[i] - excitation_wavelengths[j])
                if separation < min_sep:
                    penalty += (min_sep - separation) ** 2
        
        return penalty
    
    def multi_objective_cost(self, excitation_wavelengths: np.ndarray) -> float:
        """Multi-objective cost function combining all criteria."""
        # Convert to numpy array if needed
        wavelengths = np.asarray(excitation_wavelengths)
        
        # Calculate individual cost components
        signal_strength = self.calculate_signal_strength(wavelengths)
        crosstalk_penalty = self.calculate_crosstalk_penalty(wavelengths)
        detection_separability = self.calculate_detection_separability(wavelengths)
        background_penalty = self.calculate_background_penalty(wavelengths)
        separation_penalty = self.calculate_separation_penalty(wavelengths)
        
        # Combine with weights
        total_cost = (
            -self.config.signal_strength_weight * signal_strength +
            self.config.crosstalk_weight * crosstalk_penalty +
            -self.config.detection_separability_weight * detection_separability +
            self.config.background_weight * background_penalty +
            self.config.separation_penalty_weight * separation_penalty
        )
        
        return total_cost
    
    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds for each fluorophore."""
        bounds = []
        
        for name in self.fluorophore_names:
            if name in self.excitation_spectra:
                wl, ex = self.excitation_spectra[name]
                
                # Find peak excitation wavelength
                peak_idx = np.argmax(ex)
                peak_wl = wl[peak_idx]
                
                # Set bounds around peak
                lower = max(wl.min(), peak_wl - self.config.search_range)
                upper = min(wl.max(), peak_wl + self.config.search_range)
                
                bounds.append((lower, upper))
            else:
                # Default bounds if no spectrum available
                bounds.append((400.0, 700.0))
        
        return bounds
    
    def optimize_single_run(self, seed: int = None) -> Dict:
        """Run single optimization with given seed."""
        if seed is not None:
            np.random.seed(seed)
        
        bounds = self.get_optimization_bounds()
        
        # Run differential evolution
        result = differential_evolution(
            func=self.multi_objective_cost,
            bounds=bounds,
            strategy='best1bin',
            popsize=self.config.population_size,
            maxiter=self.config.max_iterations,
            tol=self.config.tolerance,
            seed=seed,
            polish=True
        )
        
        return {
            'wavelengths': result.x,
            'cost': result.fun,
            'success': result.success,
            'n_evaluations': result.nfev,
            'message': result.message
        }
    
    def optimize_multistart(self) -> Dict:
        """Run multi-start optimization for robustness."""
        print(f"Running multi-start optimization with {self.config.n_multistart} starts...")
        
        results = []
        seeds = np.random.randint(0, 10000, self.config.n_multistart)
        
        start_time = time.time()
        
        # Run optimizations
        for i, seed in enumerate(seeds):
            print(f"  Start {i+1}/{self.config.n_multistart}...", end=" ")
            
            try:
                result = self.optimize_single_run(seed)
                results.append(result)
                
                if result['success']:
                    print(f"Success (cost: {result['cost']:.4f})")
                else:
                    print(f"Failed: {result['message']}")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        # Find best result
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            raise RuntimeError("All optimization runs failed")
        
        best_result = min(successful_results, key=lambda x: x['cost'])
        
        # Calculate statistics
        costs = [r['cost'] for r in successful_results]
        
        return {
            'best_wavelengths': best_result['wavelengths'],
            'best_cost': best_result['cost'],
            'n_successful': len(successful_results),
            'n_total': len(results),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'all_results': results,
            'optimization_time': elapsed_time
        }
    
    def validate_solution(self, wavelengths: np.ndarray) -> Dict:
        """Validate optimization solution with detailed analysis."""
        validation = {}
        
        # Cost breakdown
        validation['signal_strength'] = self.calculate_signal_strength(wavelengths)
        validation['crosstalk_penalty'] = self.calculate_crosstalk_penalty(wavelengths)
        validation['detection_separability'] = self.calculate_detection_separability(wavelengths)
        validation['background_penalty'] = self.calculate_background_penalty(wavelengths)
        validation['separation_penalty'] = self.calculate_separation_penalty(wavelengths)
        validation['total_cost'] = self.multi_objective_cost(wavelengths)
        
        # Wavelength analysis
        separations = []
        for i in range(len(wavelengths)):
            for j in range(i + 1, len(wavelengths)):
                separations.append(abs(wavelengths[i] - wavelengths[j]))
        
        validation['min_separation'] = min(separations) if separations else 0
        validation['mean_separation'] = np.mean(separations) if separations else 0
        
        # Crosstalk matrix
        crosstalk_matrix = np.zeros((len(wavelengths), len(wavelengths)))
        for i, exc_wl_i in enumerate(wavelengths):
            for j, name_j in enumerate(self.fluorophore_names):
                if name_j in self.excitation_spectra:
                    wl, ex = self.excitation_spectra[name_j]
                    ex_interp = self._interpolate_spectrum(wl, ex)
                    wl_idx = np.argmin(np.abs(self.wavelength_grid - exc_wl_i))
                    crosstalk_matrix[i, j] = ex_interp[wl_idx]
        
        validation['crosstalk_matrix'] = crosstalk_matrix
        
        # Off-diagonal crosstalk statistics
        off_diagonal = crosstalk_matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        validation['max_crosstalk'] = off_diagonal.max()
        validation['mean_crosstalk'] = off_diagonal.mean()
        
        return validation
    
    def find_optimal_excitation(self) -> Dict[str, Union[int, float, Dict]]:
        """Main optimization function - find optimal excitation wavelengths."""
        print("=" * 60)
        print("ADVANCED EXCITATION WAVELENGTH OPTIMIZATION")
        print("=" * 60)
        print(f"Fluorophores: {', '.join(self.fluorophore_names)}")
        print(f"Search range: ±{self.config.search_range} nm around peaks")
        print(f"Min separation: {self.config.min_wavelength_separation} nm")
        
        # Run optimization
        optimization_result = self.optimize_multistart()
        
        # Validate solution
        best_wavelengths = optimization_result['best_wavelengths']
        validation = self.validate_solution(best_wavelengths)
        
        # Create results dictionary
        results = {
            'fluorophore_names': self.fluorophore_names,
            'optimal_wavelengths': dict(zip(
                self.fluorophore_names, 
                np.round(best_wavelengths).astype(int)
            )),
            'optimal_wavelengths_float': dict(zip(
                self.fluorophore_names, 
                best_wavelengths
            )),
            'optimization_result': optimization_result,
            'validation': validation,
            'config': self.config
        }
        
        # Print summary
        self._print_optimization_summary(results)
        
        return results
    
    def _print_optimization_summary(self, results: Dict):
        """Print optimization summary."""
        print(f"\nOptimization Results:")
        print(f"Success rate: {results['optimization_result']['n_successful']}/{results['optimization_result']['n_total']}")
        print(f"Best cost: {results['optimization_result']['best_cost']:.4f}")
        print(f"Optimization time: {results['optimization_result']['optimization_time']:.2f}s")
        
        print(f"\nOptimal Excitation Wavelengths:")
        for name, wl in results['optimal_wavelengths'].items():
            print(f"  {name}: {wl} nm")
        
        print(f"\nValidation Metrics:")
        val = results['validation']
        print(f"  Signal strength: {val['signal_strength']:.3f}")
        print(f"  Crosstalk penalty: {val['crosstalk_penalty']:.3f}")
        print(f"  Detection separability: {val['detection_separability']:.3f}")
        print(f"  Background penalty: {val['background_penalty']:.3f}")
        print(f"  Min wavelength separation: {val['min_separation']:.1f} nm")
        print(f"  Max crosstalk: {val['max_crosstalk']:.3f}")
        print(f"  Mean crosstalk: {val['mean_crosstalk']:.3f}")


def find_optimal_excitation_advanced(
    fluorophore_names: List[str],
    spectra_dir: Union[str, Path],
    detection_centers: Optional[List[float]] = None,
    detection_bandwidths: Optional[List[float]] = None,
    config: Optional[OptimizationConfig] = None
) -> Dict[str, int]:
    """
    Find optimal excitation wavelengths using advanced multi-objective optimization.
    
    Args:
        fluorophore_names: List of fluorophore names
        spectra_dir: Directory containing spectra files
        detection_centers: Detection channel center wavelengths
        detection_bandwidths: Detection channel bandwidths
        config: Optimization configuration
        
    Returns:
        Dictionary mapping fluorophore names to optimal excitation wavelengths
    """
    if config is None:
        config = OptimizationConfig()
    
    # Set detection configuration if provided
    if detection_centers is not None:
        config.detection_centers = detection_centers
    if detection_bandwidths is not None:
        config.detection_bandwidths = detection_bandwidths
    
    # Create optimizer and run
    optimizer = AdvancedExcitationOptimizer(fluorophore_names, spectra_dir, config)
    results = optimizer.find_optimal_excitation()
    
    return results['optimal_wavelengths']
