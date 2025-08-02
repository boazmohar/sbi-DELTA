# File: sbi_delta/simulator/emission_simulator.py

import numpy as np
from sbi_delta.simulator.base_simulator import BaseSimulator

class EmissionSimulator(BaseSimulator):
    """
    Concrete simulator using fixed filters and the new sbi_delta config/managers structure.

    Simulate detected photon counts for a batch of concentration combinations.
    counts[i, j] = photon_budget * sum_k S[i, k] * F[j, k] * Δλ
    where S is emission spectra (n_dyes×n_wl),
          F is filter_bank.filters (n_ch×n_wl),
          Δλ = config.wavelength_step.
    """

    def simulate(self, concentrations=None, add_noise=True, debug=False):
        """
        Simulate detected photon counts for each filter and excitation laser.
        Args:
            concentrations: Array of shape (n_dyes,) or (batch_size, n_dyes) or None (use default/prior)
            add_noise: Whether to add Poisson noise
            debug: If True, plot intermediate steps for inspection
        Returns:
            Array of shape (n_exc, n_channels) with detected photon counts
        """
        import numpy as np
        em = self.spectra_manager.get_emission()            # (n_dyes, n_wl)
        filt = self.filter_bank.filters                     # (n_ch,   n_wl)
        step = self.config.wavelength_step
        photon_budget = self.config.photon_budget

        # Use prior_manager to sample concentrations if not provided
        if concentrations is None:
            if hasattr(self, 'prior_manager') and self.prior_manager is not None:
                concentrations = self.prior_manager.sample()
            else:
                n_dyes = em.shape[0]
                concentrations = np.ones((n_dyes,))
        concentrations = np.asarray(concentrations)

        # Get excitation wavelengths and spectra using ExcitationManager API
        if self.excitation_manager is not None:
            exc_wavelengths = self.excitation_manager.get_wavelengths()  # (n_exc,)
            # Build excitation spectra matrix: (n_dyes, n_wl)
        # --- Setup ---
        n_dyes = len(self.config.dye_names)
        n_exc = len(self.excitation_manager.get_wavelengths())
        n_ch = len(self.filter_bank.filters)
        excitation_wavelengths = self.excitation_manager.get_wavelengths()
        wavelengths = self.spectra_manager.wavelength_grid

        # --- Emission calculation ---
        signal_matrix = np.zeros((n_exc, n_ch))
        per_dye_matrix = np.zeros((n_dyes, n_exc, n_ch))
        for i_dye, dye in enumerate(self.config.dye_names):
            emission_spectrum = self.spectra_manager.get_emission(dye)
            excitation_spectrum = self.spectra_manager.get_excitation(dye)  # (n_wl,)
            for i_exc, exc_wl in enumerate(excitation_wavelengths):
                # Find the excitation value for this dye at the current laser wavelength
                wl_grid = self.spectra_manager.wavelength_grid
                idx = np.abs(wl_grid - exc_wl).argmin()
                dye_exc = excitation_spectrum[idx]
                for i_ch, filt in enumerate(self.filter_bank.filters):
                    filter_profile = self.filter_bank[i_ch]
                    detected = np.sum(emission_spectrum * filter_profile * dye_exc)
                    per_dye_matrix[i_dye, i_exc, i_ch] = detected * concentrations[i_dye]
                    signal_matrix[i_exc, i_ch] += detected * concentrations[i_dye]

        # --- Background calculation ---
        bg_conc = 0.0
        bg_matrix = np.zeros((n_exc, n_ch))
        if self.config.bg_dye is not None:
            # Find index of bg_dye in excitation_names (for correct bg_exc extraction in plotting)
            try:
                bg_idx = self.spectra_manager.excitation_names.index(self.config.bg_dye)
            except (AttributeError, ValueError):
                bg_idx = None
            if len(concentrations) > n_dyes:
                bg_conc = concentrations[-1]
            bg_emission_spectrum = self.spectra_manager.get_emission(self.config.bg_dye)
            bg_excitation_spectrum = self.spectra_manager.get_excitation(self.config.bg_dye)
            for i_exc, exc_wl in enumerate(excitation_wavelengths):
                wl_grid = self.spectra_manager.wavelength_grid
                idx = np.abs(wl_grid - exc_wl).argmin()
                bg_exc = bg_excitation_spectrum[idx]
                for i_ch, filt in enumerate(self.filter_bank.filters):
                    filter_profile = self.filter_bank[i_ch]
                    detected = np.sum(bg_emission_spectrum * filter_profile * bg_exc)
                    bg_matrix[i_exc, i_ch] = detected

        # --- Normalization ---
        # Normalize signal by total area under curve
        signal_matrix_pre_norm = signal_matrix.copy()
        total_signal = np.sum(signal_matrix)
        if total_signal > 0:
            signal_matrix_norm = signal_matrix / total_signal * float(photon_budget)
        else:
            signal_matrix_norm = signal_matrix.copy()
        # Add background as a fraction of photon_budget, distributed by normalized bg_matrix
        signal_matrix_post_bg = signal_matrix_norm.copy()
        if bg_conc > 0:
            bg_sum = np.sum(bg_matrix)
            if bg_sum > 0:
                bg_matrix_norm = bg_matrix / bg_sum
            else:
                bg_matrix_norm = np.zeros_like(bg_matrix)
            signal_matrix_post_bg += bg_matrix_norm * float(photon_budget) * bg_conc
        # For output, use the final matrix
        signal_matrix = signal_matrix_post_bg

        # --- Noise ---
        if add_noise:
            signal_matrix = np.random.poisson(signal_matrix)

        # --- Debug plotting ---
        # --- Debug plotting ---
        if debug:
            import matplotlib.pyplot as plt
            # Heatmap before normalization
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(signal_matrix_pre_norm, aspect='auto', interpolation='none', cmap='viridis')
            ax.set_title('Photon Counts Matrix (Before Normalization)')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Excitation')
            plt.colorbar(im, ax=ax, label='Photon Count (pre-norm)')
            plt.tight_layout()
            plt.show()
            # Visualize background effect by channel (bar plot)
            if bg_conc > 0:
                # Sum background photons per channel (after normalization)
                bg_sum = np.sum(bg_matrix)
                if bg_sum > 0:
                    bg_matrix_norm = bg_matrix / bg_sum
                else:
                    bg_matrix_norm = np.zeros_like(bg_matrix)
                bg_photons_matrix = bg_matrix_norm * float(photon_budget) * bg_conc
                bg_photons_per_channel = np.sum(bg_photons_matrix, axis=0)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(np.arange(n_ch), bg_photons_per_channel, color='gray', alpha=0.7)
                ax.set_xlabel('Channel')
                ax.set_ylabel('Background Photons')
                ax.set_title('Background Photon Contribution by Channel (after normalization)')
                plt.tight_layout()
                plt.show()

            # Print normalization math
            print("--- Normalization math ---")
            print(f"total_signal = np.sum(signal_matrix_pre_norm) = {total_signal:.3f}")
            print(f"photon_budget = {photon_budget:.3f}")
            print("signal_matrix_norm = signal_matrix_pre_norm / total_signal * photon_budget")
            print("--- Matrix after normalization (before BG) ---")
            print(signal_matrix_norm)

            # Heatmap after normalization (before background)
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(signal_matrix_norm, aspect='auto', interpolation='none', cmap='viridis')
            ax.set_title('Photon Counts Matrix (After Normalization, Before BG)')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Excitation')
            plt.colorbar(im, ax=ax, label='Photon Count (norm, no BG)')
            plt.tight_layout()
            plt.show()

            # Heatmap after normalization (with background)
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(signal_matrix_post_bg, aspect='auto', interpolation='none', cmap='viridis')
            ax.set_title('Photon Counts Matrix (After Normalization + BG)')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Excitation')
            plt.colorbar(im, ax=ax, label='Photon Count (post-norm)')
            plt.tight_layout()
            plt.show()

            # Per-laser plots (existing code)
            for i_exc, exc_wl in enumerate(excitation_wavelengths):
                fig, ax = plt.subplots(figsize=(10, 5))
                # Plot excitation spectrum for each dye
                for i_dye, dye in enumerate(self.config.dye_names):
                    emission_spectrum = self.spectra_manager.get_emission(dye)
                    excitation_spectrum = self.spectra_manager.get_excitation(dye)
                    wl_grid = self.spectra_manager.wavelength_grid
                    idx = np.abs(wl_grid - exc_wl).argmin()
                    dye_exc = excitation_spectrum[idx]
                    ax.plot(wavelengths, emission_spectrum * dye_exc, label=f"{dye}")
                # Plot background if present
                if self.config.bg_dye is not None:
                    bg_emission_spectrum = self.spectra_manager.get_emission(self.config.bg_dye)
                    bg_excitation_spectrum = self.spectra_manager.get_excitation(self.config.bg_dye)
                    wl_grid = self.spectra_manager.wavelength_grid
                    idx = np.abs(wl_grid - exc_wl).argmin()
                    bg_exc = bg_excitation_spectrum[idx]
                    ax.plot(wavelengths, bg_emission_spectrum * bg_exc, label=f"{self.config.bg_dye} (bg)", linestyle=':')
                # Overlay filters
                for i_ch, filt in enumerate(self.filter_bank.filters):
                    filter_profile = self.filter_bank[i_ch]
                    ax.plot(wavelengths, filter_profile * np.max([np.max(self.spectra_manager.get_emission(d)) for d in self.config.dye_names]), '--', label=f"Filter {i_ch}")
                ax.set_title(f"Excitation {i_exc} ({exc_wl} nm)")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Emission (a.u.)")
                ax.legend(loc='upper right', fontsize='small', ncol=2)
                # Annotate photon counts for each dye and channel
                y0 = ax.get_ylim()[1]
                for i_dye, dye in enumerate(self.config.dye_names):
                    for i_ch in range(n_ch):
                        val = per_dye_matrix[i_dye, i_exc, i_ch]
                        if val > 0:
                            ax.annotate(f"{dye}→Ch{i_ch}: {val:.0f}", xy=(exc_wl, y0*0.8 - i_ch*0.05*y0 - i_dye*0.02*y0), fontsize=8, color='black')
                # Annotate background
                if bg_conc > 0:
                    for i_ch in range(n_ch):
                        val = bg_matrix[i_exc, i_ch] * float(photon_budget) * bg_conc
                        if val > 0:
                            ax.annotate(f"BG→Ch{i_ch}: {val:.0f}", xy=(exc_wl, y0*0.6 - i_ch*0.05*y0), fontsize=8, color='gray')
                plt.tight_layout()
                plt.show()
            # Print photon counts matrix
            print("Photon counts matrix (excitation x channel):")
            print(signal_matrix)
        return signal_matrix
       