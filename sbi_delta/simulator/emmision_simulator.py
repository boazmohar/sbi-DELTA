# File: sbi_delta/simulator/fixed_filter_simulator.py

import numpy as np
from sbi_delta.simulator.base_simulator import BaseSimulator

class EmmissionSimulator(BaseSimulator):
    """
    Concrete simulator using fixed filters.

    simulate() computes:
      counts[i,j] = photon_budget * sum_k S[i,k] * F[j,k] * Δλ
    where S is emission spectra (n_dyes×n_wl),
          F is filter_bank.filters (n_ch×n_wl),
          Δλ = config.wavelength_step.
    """

    def simulate(self) -> np.ndarray:
        em = self.spectra_manager.get_emission()            # (n_dyes, n_wl)
        filt = self.filter_bank.filters                     # (n_ch,   n_wl)
        step = self.config.wavelength_step
        budget = self.config.photon_budget

        # dot: (n_dyes×n_wl) · (n_wl×n_ch) = (n_dyes×n_ch)
        # note we want each dye vs each channel:
        counts = budget * (em @ filt.T) * step
        return counts
