import logging
import numpy as np
import matplotlib.pyplot as plt

from sbi_delta.config import BaseConfig, ExcitationConfig
from sbi_delta.spectra_manager import SpectraManager

logger = logging.getLogger(__name__)

class ExcitationManager:
    """
    Manages excitation wavelength selection for the simulator.

    Modes:
    - manual: manual_wavelengths provided in ExcitationConfig
    - peak: use excitation peak for each dye (requires SpectraManager)
    - min_crosstalk: grid search for minimum crosstalk (see Microscope.py logic)
    """

    def __init__(
        self,
        config: BaseConfig,
        excitation_cfg: ExcitationConfig,
        spectra_manager: SpectraManager,
    ):
        self.config = config
        self.excitation_cfg = excitation_cfg
        self.spectra_manager = spectra_manager
        self.n_dyes = len(config.dye_names)
        self.excitation_mode = excitation_cfg.excitation_mode
        self.excitation_wavelengths = self._init_wavelengths()

    def get_wavelengths(self) -> np.ndarray:
        """Return the excitation wavelengths in use."""
        return self.excitation_wavelengths

    def has_crosstalk(self) -> bool:
        """Return True if crosstalk is enabled in config."""
        return getattr(self.excitation_cfg, "include_crosstalk", False)

    def __repr__(self) -> str:
        return (f"<ExcitationManager("
                f"mode={self.excitation_mode}, "
                f"wavelengths={self.excitation_wavelengths}, "
                f"crosstalk={self.has_crosstalk()})>")

    def _find_min_crosstalk_wavelengths(self, search_ranges: list) -> np.ndarray:
        """
        Brute-force search for each dye's excitation wavelength that minimizes crosstalk,
        using the cost function from Microscope.py. No evolutionary optimization, just grid search.
        search_ranges: list of (min, max) tuples, one per dye.
        Returns: np.ndarray of optimal wavelengths (one per dye)
        """
        spectra_dict = {}
        # Add all main dyes
        for dye in self.config.dye_names:
            wl = self.spectra_manager.wavelength_grid
            ex = self.spectra_manager.get_excitation(dye)
            spectra_dict[dye] = (wl, ex)
        # Add background dye if present and loaded
        bg_dye = getattr(self.config, "bg_dye", None)
        include_bg = bg_dye is not None and bg_dye in self.spectra_manager.excitation_names
        if include_bg:
            wl = self.spectra_manager.wavelength_grid
            ex = self.spectra_manager.get_excitation(bg_dye)
            spectra_dict[bg_dye] = (wl, ex)

        shared_wl = self.spectra_manager.wavelength_grid

        # Only optimize main dyes, but include bg in cost
        fluor_names = list(self.config.dye_names)
        best_wavelengths = []
        for i, dye in enumerate(fluor_names):
            min_wl, max_wl = search_ranges[i]
            candidates = np.arange(int(np.ceil(min_wl)), int(np.floor(max_wl)) + 1)
            best_cost = None
            best_wl = None
            peak = shared_wl[np.argmax(spectra_dict[dye][1])]
            for wl in candidates:
                candidate_wavelengths = []
                # Main dyes: optimize their wavelength
                for j, d in enumerate(fluor_names):
                    if i == j:
                        candidate_wavelengths.append(wl)
                    else:
                        ex = spectra_dict[d][1]
                        idx = np.argmax(ex)
                        candidate_wavelengths.append(shared_wl[idx])
                # Add background dye's excitation peak if present
                if include_bg:
                    ex_bg = spectra_dict[bg_dye][1]
                    idx_bg = np.argmax(ex_bg)
                    bg_peak = shared_wl[idx_bg]
                    candidate_wavelengths.append(bg_peak)
                # Names for cost: main dyes + bg if present
                cost_names = fluor_names + ([bg_dye] if include_bg else [])
                cost = self._excitation_cost(
                    np.array(candidate_wavelengths),
                    {k: spectra_dict[k] for k in cost_names},
                    shared_wl, verbose=True
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_wl = wl
                elif cost == best_cost:
                    # Prefer the candidate closest to the peak
                    if abs(wl - peak) < abs(best_wl - peak):
                        best_wl = wl
            best_wavelengths.append(best_wl)
        return np.array(best_wavelengths)

    @staticmethod
    def _excitation_cost(wavelengths, spectra_dict, shared_wl, verbose=False):
        """
        Cost function: maximize self-signal, minimize crosstalk.
        Includes background dye if present in spectra_dict/wavelengths.
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

            total_cost += -self_signal + crosstalk
        return total_cost

    def _init_wavelengths(self) -> np.ndarray:
        exc_cfg: ExcitationConfig = self.excitation_cfg
        if self.excitation_mode == "manual":
            if exc_cfg.manual_wavelengths is None:
                raise ValueError("Manual excitation mode requires manual_wavelengths in ExcitationConfig.")
            wl = np.array(exc_cfg.manual_wavelengths)
            if len(wl) != self.n_dyes:
                raise ValueError(f"manual_wavelengths length ({len(wl)}) does not match n_dyes ({self.n_dyes})")
            return wl
        elif self.excitation_mode == "peak":
            peaks = []
            for dye in self.config.dye_names:
                spectrum = self.spectra_manager.get_excitation(dye)
                idx = np.argmax(spectrum)
                peak_wavelength = self.spectra_manager.wavelength_grid[idx]
                peaks.append(peak_wavelength)
            return np.array(peaks)
        elif self.excitation_mode == "min_crosstalk":
            # Use search_range from ExcitationConfig or default to ±30nm around peak
            search_ranges = []
            for dye in self.config.dye_names:
                ex = self.spectra_manager.get_excitation(dye)
                wl = self.spectra_manager.wavelength_grid
                idx = np.argmax(ex)
                peak = wl[idx]
                if self.excitation_cfg.search_range is not None:
                    min_wl, max_wl = self.excitation_cfg.search_range
                else:
                    min_wl = max(wl[0], peak - 30)
                    max_wl = min(wl[-1], peak + 30)
                search_ranges.append((min_wl, max_wl))
            return self._find_min_crosstalk_wavelengths(search_ranges)
        else:
            raise ValueError(f"Unknown excitation_mode:{self.excitation_mode}")

    def plot_excitation_wavelengths(self, ax=None) -> "plt.Axes":
        """
        Plot vertical lines for each excitation wavelength.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for dye, wl in zip(self.config.dye_names, self.get_wavelengths()):
            ax.axvline(wl, linestyle='--', label=f"{dye} excitation @ {wl:.1f} nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Excitation event")
        ax.set_title("Excitation Wavelengths")
        ax.legend()
        return ax

    def plot_on_spectra(self, spectra_manager, ax=None) -> "plt.Axes":
        """
        Overlay excitation wavelengths on top of all excitation spectra.
        """
        if ax is None:
            fig, ax = plt.subplots()
        spectra_manager.plot_all_excitations(ax=ax)
        for dye, wl in zip(self.config.dye_names, self.get_wavelengths()):
            ax.axvline(wl, linestyle=':', color='black', alpha=0.7, label=f"{dye} λ={wl:.1f} nm")
                # Plot the excitation spectrum for the background dye
        bg_dye = self.config.bg_dye
        if bg_dye is not None and bg_dye in spectra_manager.excitation_names:
            ax = spectra_manager.plot_excitation(bg_dye)
            ax.set_title(f"Excitation Spectrum: {bg_dye} (Background)")
            plt.show()
        else:
            print("Background dye excitation not found or not loaded.")
        ax.legend()
        return ax

    def print_assignment_table(self):
        """
        Print a simple table of dye to excitation wavelength assignment.
        """
        print("Excitation wavelength assignment:")
        for dye, wl in zip(self.config.dye_names, self.get_wavelengths()):
            print(f"  {dye}: {wl:.1f} nm")
    def __repr__(self) -> str:
        return (
            f"<ExcitationManager("
            f"mode={self.excitation_mode}, "
            f"wavelengths={self.excitation_wavelengths}, "
            f"crosstalk={self.has_crosstalk()})>"
        )

    def plot_crosstalk_matrix(self, ax=None, annotate=True, include_bg=True) -> "plt.Axes":
        """
        Plot a matrix showing crosstalk (excitation at each dye's wavelength for all dyes),
        including the background dye if present and loaded.
        """
        import matplotlib.pyplot as plt

        dye_names = list(self.config.dye_names)
        wavelengths = list(self.get_wavelengths())

        # Optionally include background dye
        if include_bg and getattr(self.config, "bg_dye", None):
            bg_dye = self.config.bg_dye
            if bg_dye in self.spectra_manager.excitation_names:
                dye_names.append(bg_dye)
                # For background, use the same excitation selection logic (peak or min_crosstalk)
                ex = self.spectra_manager.get_excitation(bg_dye)
                wl = self.spectra_manager.wavelength_grid
                idx = np.argmax(ex)
                bg_peak = wl[idx]
                wavelengths.append(bg_peak)

        n = len(dye_names)
        matrix = np.zeros((n, n))
        shared_wl = self.spectra_manager.wavelength_grid

        for i, dye_i in enumerate(dye_names):
            ex_i = self.spectra_manager.get_excitation(dye_i)
            for j, wl in enumerate(wavelengths):
                idx = np.clip(np.searchsorted(shared_wl, wl, side="left"), 0, len(ex_i) - 1)
                matrix[i, j] = ex_i[idx]

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([f"{d} ({w:.0f})" for d, w in zip(dye_names, wavelengths)], rotation=45, ha="right")
        ax.set_yticklabels(dye_names)
        ax.set_xlabel("Excitation wavelength (nm)")
        ax.set_ylabel("Dye (excited)")
        ax.set_title("Excitation Crosstalk Matrix")
        plt.colorbar(im, ax=ax, label="Normalized Excitation")
        if annotate:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w" if matrix[i, j] < 0.5 else "black")
        return ax