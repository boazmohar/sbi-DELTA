# File: sbi_delta/simulator/variable_emission_simulator.py

import numpy as np
from scipy.interpolate import interp1d
from sbi_delta.simulator.base_simulator import BaseSimulator
from sbi_delta.config import FilterConfig
from sbi_delta.filter_bank import FilterBank

class VariableEmissionSimulator(BaseSimulator):
    """
    Variable emission simulator with hierarchical adaptive filtering.
    
    This simulator starts with initial broad detection filters from the provided FilterBank
    and recursively subdivides them into 3 narrower filters if the summed photon count 
    across all excitation lasers exceeds a threshold, until filters reach minimum width.
    Uses FilterBank to create and manage filters properly.
    """
    
    def __init__(self, spectra_manager, filter_bank, config, excitation_manager=None, prior_manager=None, 
                 threshold=20, min_filter_width=10):
        """
        Initialize the variable emission simulator.
        
        Args:
            spectra_manager: SpectraManager object for loading fluorophore spectra (must have .load() called)
            filter_bank: FilterBank object containing initial broad filters to start subdivision from
            config: BaseConfig object containing simulation parameters
            excitation_manager: ExcitationManager object for managing excitation wavelengths (optional)
            prior_manager: PriorManager object for sampling concentrations (optional)
            threshold: Photon count threshold above which filters are subdivided
            min_filter_width: Minimum filter width (nm) for subdivision stopping criterion
        """
        super().__init__(spectra_manager, filter_bank, config, excitation_manager, prior_manager)
        self.threshold = threshold
        self.min_filter_width = min_filter_width
        
        # Extract sharpness from the provided FilterBank (we'll create our own filters per laser)
        self.filter_sharpness = 0.1  # Default fallback
        if filter_bank.configs:
            self.filter_sharpness = filter_bank.configs[0].sharpness
        
        # We'll create laser-specific filters dynamically in simulate()
    
    def _create_filter_bank(self, filter_ranges):
        """
        Create a FilterBank from filter ranges using FilterConfig objects.
        
        Args:
            filter_ranges: List of (start, stop) wavelength tuples
            
        Returns:
            FilterBank object
        """
        filter_configs = [
            FilterConfig(start=start, stop=stop, sharpness=self.filter_sharpness)
            for start, stop in filter_ranges
        ]
        return FilterBank(self.config, filter_configs)
    
    def _calculate_photon_counts_with_filter_bank(self, concentrations, filter_bank, excitation_wavelengths):
        """
        Calculate photon counts using a FilterBank (similar to EmissionSimulator logic).
        
        Args:
            concentrations: Array of fluorophore concentrations
            filter_bank: FilterBank object with filters
            excitation_wavelengths: List of excitation wavelengths
            
        Returns:
            Array of shape (n_exc, n_filters) with photon counts
        """
        wavelengths = self.spectra_manager.wavelength_grid
        n_dyes = len(self.config.dye_names)
        n_exc = len(excitation_wavelengths)
        n_filters = len(filter_bank.filters)
        
        # Initialize matrices
        signal_matrix = np.zeros((n_exc, n_filters))
        
        # --- Emission calculation (similar to EmissionSimulator) ---
        for i_dye, dye in enumerate(self.config.dye_names):
            emission_spectrum = self.spectra_manager.get_emission(dye)
            excitation_spectrum = self.spectra_manager.get_excitation(dye)
            
            for i_exc, exc_wl in enumerate(excitation_wavelengths):
                # Find excitation efficiency at this wavelength
                idx = np.abs(wavelengths - exc_wl).argmin()
                dye_exc = excitation_spectrum[idx]
                
                for i_filt in range(n_filters):
                    filter_profile = filter_bank[i_filt]  # Use FilterBank indexing
                    detected = np.sum(emission_spectrum * filter_profile * dye_exc * self.config.wavelength_step)
                    signal_matrix[i_exc, i_filt] += detected * concentrations[i_dye]
        
        # --- Background calculation ---
        if self.config.bg_dye is not None:
            bg_conc = concentrations[-1] if len(concentrations) > n_dyes else 0.0
            if bg_conc > 0:
                bg_emission_spectrum = self.spectra_manager.get_emission(self.config.bg_dye)
                bg_excitation_spectrum = self.spectra_manager.get_excitation(self.config.bg_dye)
                
                for i_exc, exc_wl in enumerate(excitation_wavelengths):
                    idx = np.abs(wavelengths - exc_wl).argmin()
                    bg_exc = bg_excitation_spectrum[idx]
                    
                    for i_filt in range(n_filters):
                        filter_profile = filter_bank[i_filt]
                        detected = np.sum(bg_emission_spectrum * filter_profile * bg_exc * self.config.wavelength_step)
                        signal_matrix[i_exc, i_filt] += detected * bg_conc
        
        return signal_matrix
    
    def _subdivide_filter_range(self, start_wl, stop_wl):
        """
        Subdivide a filter range into 3 equal parts.
        
        Args:
            start_wl: Start wavelength
            stop_wl: Stop wavelength
            
        Returns:
            List of 3 (start, stop) tuples
        """
        range_width = (stop_wl - start_wl) / 3
        return [
            (start_wl, start_wl + range_width),
            (start_wl + range_width, start_wl + 2 * range_width),
            (start_wl + 2 * range_width, stop_wl)
        ]
    
    def _recursive_subdivision_per_laser(self, concentrations, exc_wl, filter_ranges, 
                                        normalization_factor, measurements_dict, debug=False):
        """
        Recursively subdivide filters for a specific laser based on photon count threshold.
        
        Args:
            concentrations: Fluorophore concentrations
            exc_wl: Specific excitation wavelength
            filter_ranges: Current list of (start, stop) filter ranges for this laser
            normalization_factor: Normalization factor from initial broad measurements
            measurements_dict: Dictionary to store measurements
            debug: Print debug information
        """
        # Create FilterBank for current ranges
        current_filter_bank = self._create_filter_bank(filter_ranges)
        
        # Calculate photon counts for this laser only
        counts = self._calculate_photon_counts_with_filter_bank(
            concentrations, current_filter_bank, [exc_wl]  # Single laser
        )
        
        # Apply normalization
        counts *= normalization_factor
        
        # Check each filter for subdivision
        new_ranges_to_process = []
        
        for i_filt, (start_wl, stop_wl) in enumerate(filter_ranges):
            center_wl = (start_wl + stop_wl) / 2
            
            # Get photon count for this laser-filter combination
            photon_count = counts[0, i_filt]  # counts shape is (1, n_filters) since single laser
            
            # Store measurement
            measurements_dict[(exc_wl, center_wl)] = photon_count
            
            if debug:
                print(f"  Filter {start_wl:.0f}-{stop_wl:.0f}nm: {photon_count:.1f} photons")
            
            # Check if subdivision is needed (per-laser threshold check)
            filter_width = stop_wl - start_wl
            should_subdivide = (photon_count > self.threshold and filter_width > self.min_filter_width)
            
            if should_subdivide:
                if debug:
                    print(f"    -> Subdividing (count={photon_count:.1f} > {self.threshold}, width={filter_width:.0f} > {self.min_filter_width})")
                
                sub_ranges = self._subdivide_filter_range(start_wl, stop_wl)
                new_ranges_to_process.extend(sub_ranges)
            else:
                if debug:
                    reason = f"count={photon_count:.1f} <= {self.threshold}" if photon_count <= self.threshold else f"width={filter_width:.0f} <= {self.min_filter_width}"
                    print(f"    -> No subdivision ({reason})")
        
        # Recursively process subdivided ranges for this laser
        if new_ranges_to_process:
            if debug:
                print(f"  Recursively processing {len(new_ranges_to_process)} subdivided ranges for laser {exc_wl}nm")
            self._recursive_subdivision_per_laser(
                concentrations, exc_wl, new_ranges_to_process, 
                normalization_factor, measurements_dict, debug
            )
    
    def simulate(self, concentrations=None, add_noise=True, debug=False):
        """
        Simulate detected photon counts using hierarchical adaptive filtering.
        
        Args:
            concentrations: Array of shape (n_dyes,) or (batch_size, n_dyes) or None
            add_noise: Whether to add Poisson noise
            debug: If True, print debug information
            
        Returns:
            Dictionary containing:
                - 'measurements': Dict mapping (excitation_wl, detection_wl) -> photon_count
                - 'interpolated_spectrum': Array of shape (n_exc, n_interp_points) 
                - 'wavelength_grid': Wavelength grid for interpolated spectrum (10nm resolution)
        """
        # Get concentrations
        if concentrations is None:
            if hasattr(self, 'prior_manager') and self.prior_manager is not None:
                concentrations = self.prior_manager.sample()
            else:
                n_dyes = len(self.config.dye_names)
                concentrations = np.ones((n_dyes,))
        concentrations = np.asarray(concentrations)
        
        # Get excitation wavelengths
        excitation_wavelengths = self.excitation_manager.get_wavelengths()
        
        if debug:
            print(f"Starting simulation with {len(self.config.dye_names)} dyes")
            print(f"Excitation wavelengths: {excitation_wavelengths}")
            print(f"Threshold for subdivision: {self.threshold}")
            print(f"Minimum filter width: {self.min_filter_width} nm")
        
        # Phase 1: Establish normalization using laser-specific broad filters (3 per laser)
        if debug:
            print("\n=== Phase 1: Calculate normalization factor ===")
        
        # Process each laser separately and build normalization matrix
        all_broad_counts = []
        total_signal = 0.0
        
        for i, exc_wl in enumerate(excitation_wavelengths):
            if debug:
                print(f"\nLaser {exc_wl}nm broad filters:")
            
            # Divide the range from laser wavelength to max wavelength into 3 parts
            range_width = (self.config.max_wavelength - exc_wl) / 3
            laser_filters = [
                (exc_wl, exc_wl + range_width),
                (exc_wl + range_width, exc_wl + 2 * range_width), 
                (exc_wl + 2 * range_width, self.config.max_wavelength)
            ]
            
            if debug:
                for j, (start, stop) in enumerate(laser_filters):
                    print(f"  Filter {j+1}: {start:.0f}-{stop:.0f} nm")
            
            # Create FilterBank with only this laser's 3 filters
            laser_filter_bank = self._create_filter_bank(laser_filters)
            
            # Calculate photon counts for this laser only with its filters
            laser_counts = self._calculate_photon_counts_with_filter_bank(
                concentrations, laser_filter_bank, [exc_wl]  # Only this laser
            )
            
            # laser_counts shape is (1, 3) - one laser, three filters
            laser_counts = laser_counts[0, :]  # Shape becomes (3,)
            all_broad_counts.append(laser_counts)
            total_signal += np.sum(laser_counts)
            
            if debug:
                print(f"  Photon counts: {laser_counts}")
        
        # Normalization (done only once as requested)
        if total_signal > 0:
            normalization_factor = self.config.photon_budget / total_signal
        else:
            normalization_factor = 1.0
        
        if debug:
            print(f"\nTotal signal before normalization: {total_signal:.3f}")
            print(f"Normalization factor: {normalization_factor:.3f}")
        
        # Phase 2: Process each laser independently
        if debug:
            print(f"\n=== Phase 2: Independent laser processing ===")
        
        all_measurements = {}
        
        # Process each laser independently
        for exc_wl in excitation_wavelengths:
            if debug:
                print(f"\n--- Processing Laser {exc_wl} nm ---")
            
            # Create initial filter range for this laser: start 10nm after laser, go to max wavelength
            laser_start = exc_wl + 10
            laser_filter_ranges = [(laser_start, self.config.max_wavelength)]
            
            if debug:
                print(f"Initial filter range for laser {exc_wl}nm: {laser_start}-{self.config.max_wavelength} nm")
            
            # Start recursive subdivision for this laser
            self._recursive_subdivision_per_laser(
                concentrations, exc_wl, laser_filter_ranges,
                normalization_factor, all_measurements, debug
            )
        
        # Phase 3: Add noise if requested
        if add_noise:
            for key in all_measurements:
                all_measurements[key] = np.random.poisson(max(0, all_measurements[key]))
        
        # Phase 4: Interpolate to 10nm resolution
        if debug:
            print(f"\n=== Phase 3: Interpolating to 10nm resolution ===")
        
        # Create 10nm wavelength grid
        min_wl = self.config.min_wavelength
        max_wl = self.config.max_wavelength  
        interp_wavelengths = np.arange(min_wl, max_wl + 10, 10)
        
        # Interpolate for each excitation wavelength
        interpolated_spectra = np.zeros((len(excitation_wavelengths), len(interp_wavelengths)))
        
        for i_exc, exc_wl in enumerate(excitation_wavelengths):
            # Get all detection wavelengths and counts for this excitation
            det_wavelengths = []
            det_counts = []
            
            for (exc, det), count in all_measurements.items():
                if exc == exc_wl:
                    det_wavelengths.append(det)
                    det_counts.append(count)
            
            if len(det_wavelengths) > 1:
                # Sort by wavelength
                sorted_indices = np.argsort(det_wavelengths)
                det_wavelengths = np.array(det_wavelengths)[sorted_indices]
                det_counts = np.array(det_counts)[sorted_indices]
                
                # Interpolate with extrapolation to 0 outside measurement range
                # Use configurable interpolation method from config
                interp_func = interp1d(det_wavelengths, det_counts, kind=self.config.interpolation_kind, 
                                     bounds_error=False, fill_value=0.0)
                interpolated_spectra[i_exc, :] = interp_func(interp_wavelengths)
            elif len(det_wavelengths) == 1:
                # Only one measurement point - set only that wavelength, rest stays 0
                det_wl = det_wavelengths[0]
                det_count = det_counts[0]
                # Find closest wavelength in interpolation grid
                closest_idx = np.abs(interp_wavelengths - det_wl).argmin()
                interpolated_spectra[i_exc, closest_idx] = det_count
                # All other wavelengths remain 0 (already initialized)
            # If no measurements for this laser, spectrum remains all zeros
            
            if debug:
                print(f"Excitation {exc_wl}nm: interpolated from {len(det_wavelengths)} points")
        
        if debug:
            print(f"\nFinal interpolated spectrum shape: {interpolated_spectra.shape}")
            print(f"Wavelength grid: {interp_wavelengths[0]:.0f} to {interp_wavelengths[-1]:.0f} nm ({len(interp_wavelengths)} points)")
            print(f"Total measurements taken: {len(all_measurements)}")
            
            # Return comprehensive results for debugging
            return {
                'measurements': all_measurements,
                'interpolated_spectrum': interpolated_spectra,
                'wavelength_grid': interp_wavelengths,
                'broad_filter_counts': all_broad_counts,
                'normalization_factor': normalization_factor
            }
        else:
            # Return numpy array for SBI framework
            # Shape: (n_excitations, n_wavelength_bins_10nm)
            # Transpose to match expected format: (n_wavelength_bins_10nm, n_excitations)
            return interpolated_spectra.T