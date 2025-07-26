"""
SpectraManager: simple load from a single folder of NPZ files for emission and excitation
using keys 'wavelengths_emission'/'emission' and 'wavelengths_excitation'/'excitation', with logging.
"""
import numpy as np
import logging
from pathlib import Path
from typing import Sequence, Union, List, Optional
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectraManager:
    """
    Load, peak-normalise, and interpolate emission and excitation spectra by dye name
    from a single folder of NPZ files where each file <dye>.npz contains four arrays:
      'wavelengths_emission', 'emission', 'wavelengths_excitation', 'excitation'.

    Parameters
    ----------
    config : BaseConfig
        Contains min_wavelength, max_wavelength, wavelength_step.
    spectra_folder : Union[str, Path]
        Folder containing one NPZ per dye (named <dye>.npz).
    dye_names : Sequence[str]
        List of dye stems to load (e.g. ['AF488', 'AF647']).

    Attributes after load:
      wavelength_grid : np.ndarray
      emission_names : List[str]
      excitation_names : List[str]
      _emission : np.ndarray  # shape (n_dyes, n_wavelengths)
      _excitation : np.ndarray # shape (n_dyes, n_wavelengths)
    """

   
    def __init__(self, config):
        self.config = config
        self.spectra_folder = Path(config.spectra_folder)
        self.dye_names = list(config.dye_names)
        self.bg_dye = config.bg_dye
        self.emission_names: List[str] = []
        self.excitation_names: List[str] = []
        self._emission: Optional[np.ndarray] = None
        self._excitation: Optional[np.ndarray] = None
        self.wavelength_grid = np.arange(
            config.min_wavelength,
            config.max_wavelength + config.wavelength_step,
            config.wavelength_step,
        )
        logger.info(f"Initialized SpectraManager(folder={self.spectra_folder}, dyes={self.dye_names}, bg_dye={self.bg_dye})")

    @staticmethod
    def list_npz(folder: Union[str, Path]) -> List[Path]:
        """Return sorted list of .npz files in the given folder."""
        p = Path(folder)
        if not p.is_dir():
            logger.error(f"list_npz: '{folder}' is not a directory")
            raise ValueError(f"'{folder}' is not a directory")
        files = sorted(p.glob("*.npz"))
        logger.info(f"Found {len(files)} .npz files in '{folder}'")
        return files

    def load(self) -> None:
        """Load, peak-normalise and interpolate spectra for each dye, including bg_dye if not in dye_names."""
        logger.info("Starting load() of spectra")
        files = {f.stem: f for f in self.list_npz(self.spectra_folder)}
        em_data: List[np.ndarray] = []
        ex_data: List[np.ndarray] = []
        self.emission_names = []
        self.excitation_names = []

        # Always include bg_dye if specified and not already in dye_names
        dyes_to_load = list(self.dye_names)
        if self.bg_dye and self.bg_dye not in dyes_to_load:
            dyes_to_load.append(self.bg_dye)

        def _process(
            name: str,
            path: Path,
            wl_key: str,
            int_key: str,
            store_list: List[np.ndarray],
            name_list: List[str],
            kind: str
        ):
            logger.info(f"Loading {kind} spectrum for dye '{name}' from {path}")
            arr = np.load(path)
            logger.debug(f"NPZ keys for '{name}': {arr.files}")
            try:
                wl = arr[wl_key]
                intensity = arr[int_key]
            except KeyError:
                logger.error(
                    f"NPZ for dye '{name}' missing '{wl_key}' or '{int_key}'; available keys: {arr.files}"
                )
                raise
            # sort
            idx = np.argsort(wl)
            wl = wl[idx]
            intensity = intensity[idx]
            # peak-normalise
            peak = float(np.max(intensity))
            if peak <= 0:
                logger.error(f"Non-positive peak ({peak}) for '{name}'")
                raise ValueError(f"Spectrum '{name}' has non-positive peak")
            intensity = intensity / peak
            # interpolate
            interp = np.interp(self.wavelength_grid, wl, intensity, left=0.0, right=0.0)
            store_list.append(interp)
            name_list.append(name)
            logger.info(f"Completed processing for '{name}'")

        # process emission and excitation using fixed keys
        for dye in dyes_to_load:
            path = files.get(dye)
            if path:
                _process(dye, path, 'wavelengths_emission', 'emission', em_data, self.emission_names, 'emission')
            else:
                logger.warning(f"Emission file for dye '{dye}' not found")

        for dye in dyes_to_load:
            path = files.get(dye)
            if path:
                _process(dye, path, 'wavelengths_excitation', 'excitation', ex_data, self.excitation_names, 'excitation')
            else:
                logger.warning(f"Excitation file for dye '{dye}' not found")

        self._emission = np.vstack(em_data) if em_data else np.empty((0, len(self.wavelength_grid)))
        self._excitation = np.vstack(ex_data) if ex_data else np.empty((0, len(self.wavelength_grid)))
        logger.info(
            f"Completed load(): {len(self.emission_names)} emission dyes, {len(self.excitation_names)} excitation dyes"
        )

    def get_emission(self, name: Optional[str] = None) -> np.ndarray:
        """Return the full emission matrix or the row for a specific dye."""
        if self._emission is None:
            logger.error("get_emission called before load()")
            raise RuntimeError("Call load() before accessing emission spectra")
        if name is None:
            return self._emission
        if name not in self.emission_names:
            logger.error(f"Emission dye '{name}' not loaded")
            raise KeyError(f"Emission dye '{name}' not loaded")
        idx = self.emission_names.index(name)
        return self._emission[idx]

    def get_excitation(self, name: Optional[str] = None) -> np.ndarray:
        """Return the full excitation matrix or the row for a specific dye."""
        if self._excitation is None:
            logger.error("get_excitation called before load()")
            raise RuntimeError("Call load() before accessing excitation spectra")
        if name is None:
            return self._excitation
        if name not in self.excitation_names:
            logger.error(f"Excitation dye '{name}' not loaded")
            raise KeyError(f"Excitation dye '{name}' not loaded")
        idx = self.excitation_names.index(name)
        return self._excitation[idx]
    
    def get_bg_emission(self) -> np.ndarray:
        """Return emission spectrum for background dye (autofluorescence)."""
        if self.bg_dye is None:
            raise ValueError("No bg_dye specified in config")
        return self.get_emission(self.bg_dye)

    def get_bg_excitation(self) -> np.ndarray:
        """Return excitation spectrum for background dye (autofluorescence)."""
        if self.bg_dye is None:
            raise ValueError("No bg_dye specified in config")
        return self.get_excitation(self.bg_dye)

    def __repr__(self) -> str:
        bg = f"\n BG dye: {self.bg_dye}" if self.bg_dye else ""
        return (
            f"<SpectraManager(folder={self.spectra_folder!r}, dyes={self.dye_names!r}){bg}\n"
            f" Emission loaded: {self.emission_names}\n"
            f" Excitation loaded: {self.excitation_names}>"
        )
    
    def plot_emission(self, dye: str, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Plot a single dye’s emission spectrum."""
        if ax is None:
            fig, ax = plt.subplots()
        wl = self.wavelength_grid
        y = self.get_emission(dye)
        ax.plot(wl, y, label=f"Emission: {dye}", linestyle='-')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Emission Spectrum: {dye}')
        ax.legend()
        return ax

    def plot_excitation(self, dye: str, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Plot a single dye’s excitation spectrum."""
        if ax is None:
            fig, ax = plt.subplots()
        wl = self.wavelength_grid
        y = self.get_excitation(dye)
        ax.plot(wl, y, label=f"Excitation: {dye}", linestyle='--')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Excitation Spectrum: {dye}')
        ax.legend()
        return ax

    def plot_all_emissions(self, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Overlay all emission spectra."""
        if ax is None:
            fig, ax = plt.subplots()
        for dye in self.emission_names:
            y = self.get_emission(dye)
            style = {}
            if self.bg_dye and dye == self.bg_dye:
                style = dict(color='gray')
                label = f"BG Emission: {dye}"
            else:
                label = f"Emission: {dye}"
            ax.plot(self.wavelength_grid, y, label=label, linestyle='-', **style)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('All Emission Spectra')
        ax.legend()
        return ax

    def plot_all_excitations(self, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Overlay all excitation spectra."""
        if ax is None:
            fig, ax = plt.subplots()
        for dye in self.excitation_names:
            y = self.get_excitation(dye)
            style = {}
            if self.bg_dye and dye == self.bg_dye:
                style = dict(color='gray')
                label = f"BG Excitation: {dye}"
            else:
                label = f"Excitation: {dye}"
            ax.plot(self.wavelength_grid, y, label=label, linestyle='--', **style)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('All Excitation Spectra')
        ax.legend()
        return ax

    def plot_combined(self, dye: str, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Plot both emission and excitation for a single dye, with distinct styles."""
        if ax is None:
            fig, ax = plt.subplots()
        wl = self.wavelength_grid
        em = self.get_emission(dye)
        ex = self.get_excitation(dye)
        em_style = {'linestyle': '-', 'label': f'Emission: {dye}'}
        ex_style = {'linestyle': '--', 'label': f'Excitation: {dye}'}
        if self.bg_dye and dye == self.bg_dye:
            em_style['color'] = 'gray'
            em_style['label'] = f'BG Emission: {dye}'
            ex_style['color'] = 'gray'
            ex_style['label'] = f'BG Excitation: {dye}'
        ax.plot(wl, em, **em_style)
        ax.plot(wl, ex, **ex_style)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Emission & Excitation: {dye}')
        ax.legend()
        return ax

    def plot_all_spectra(self, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Overlay all emission and excitation spectra across all dyes, labeling background distinctly."""
        if ax is None:
            fig, ax = plt.subplots()
        # Plot all regular dyes
        for dye in self.dye_names:
            if dye in self.emission_names:
                ax.plot(self.wavelength_grid, self.get_emission(dye), label=f'Em: {dye}', linestyle='-')
            if dye in self.excitation_names:
                ax.plot(self.wavelength_grid, self.get_excitation(dye), label=f'Ex: {dye}', linestyle='--')
        # Plot background dye if present and loaded
        if self.bg_dye:
            if self.bg_dye in self.emission_names:
                ax.plot(self.wavelength_grid, self.get_emission(self.bg_dye),
                        label=f'BG Emission: {self.bg_dye}', linestyle='-', color='gray')
            if self.bg_dye in self.excitation_names:
                ax.plot(self.wavelength_grid, self.get_excitation(self.bg_dye),
                        label=f'BG Excitation: {self.bg_dye}', linestyle='--', color='gray')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('All Emission & Excitation Spectra')
        ax.legend()
        return ax

    def plot_bg_emission(self, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Plot the background dye’s emission spectrum."""
        if self.bg_dye is None:
            raise ValueError("No bg_dye specified in config")
        if ax is None:
            fig, ax = plt.subplots()
        wl = self.wavelength_grid
        y = self.get_bg_emission()
        ax.plot(wl, y, label=f"BG Emission: {self.bg_dye}", linestyle="-", color="gray")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Emission Spectrum: {self.bg_dye} (Background)')
        ax.legend()
        return ax

    def plot_bg_excitation(self, ax: Optional["plt.Axes"] = None) -> "plt.Axes":
        """Plot the background dye’s excitation spectrum."""
        if self.bg_dye is None:
            raise ValueError("No bg_dye specified in config")
        if ax is None:
            fig, ax = plt.subplots()
        wl = self.wavelength_grid
        y = self.get_bg_excitation()
        ax.plot(wl, y, label=f"BG Excitation: {self.bg_dye}", linestyle="--", color="gray")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Excitation Spectrum: {self.bg_dye} (Background)')