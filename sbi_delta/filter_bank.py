# File: sbi_delta/filter_bank.py
"""
FilterBank for SBI-DELTA:
  - uses BaseConfig to define the wavelength grid
  - accepts multiple FilterConfig(start, stop, sharpness)
  - enforces sorted, non-overlapping filters
  - precomputes filter curves on init & on config change
  - offers repr, indexing, and plotting methods
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Sequence, Optional
from sbi_delta.config import BaseConfig, FilterConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class FilterBank:
    """
    Builds and stores bandpass filters defined by start/stop/sharpness.

    Parameters
    ----------
    base_config : BaseConfig
        Defines min/max/step for the wavelength grid.
    configs : Sequence[FilterConfig]
        One FilterConfig per channel.
    """
    def __init__(self,
                 base_config: BaseConfig,
                 configs: Sequence[FilterConfig]):
        # 1) wavelength grid
        self.wavelength_grid = np.arange(
            base_config.min_wavelength,
            base_config.max_wavelength + base_config.wavelength_step,
            base_config.wavelength_step,
        )
        logger.info(f"FilterBank: grid {self.wavelength_grid[0]}–"
                    f"{self.wavelength_grid[-1]} nm @ "
                    f"{base_config.wavelength_step} nm step")

        # 2) set and validate configs
        self.set_filters(configs)

    def set_filters(self, configs: Sequence[FilterConfig]) -> None:
        """
        Replace the current filter configs with a new list,
        sort them, check for overlaps, and rebuild the curves.
        """
        # sort
        sorted_cfg = sorted(configs, key=lambda c: c.start)
        # check overlaps
        for i in range(len(sorted_cfg) - 1):
            if sorted_cfg[i].stop > sorted_cfg[i+1].start:
                raise ValueError(
                    f"Filter overlap: channel {i} stop={sorted_cfg[i].stop} "
                    f"> channel {i+1} start={sorted_cfg[i+1].start}"
                )
        self.configs = sorted_cfg
        logger.info(f"FilterBank: loaded {len(self.configs)} filters")
        # rebuild curves
        self._build_filters()

    def _build_filters(self) -> None:
        """Compute and store the filter curves matrix (n_filters × n_wavelengths)."""
        n_ch = len(self.configs)
        n_wl = self.wavelength_grid.size
        self.filters = np.zeros((n_ch, n_wl), dtype=float)
        for i, cfg in enumerate(self.configs):
            s, e, sh = cfg.start, cfg.stop, cfg.sharpness
            wl = self.wavelength_grid
            f_low  = 1.0 / (1.0 + np.exp(-(wl - s)/sh))
            f_high = 1.0 / (1.0 + np.exp((wl - e)/sh))
            self.filters[i] = f_low * f_high
            logger.debug(f"Filter {i}: start={s}, stop={e}, sharp={sh}, max={self.filters[i].max():.3f}")

    def __repr__(self) -> str:
        specs = ", ".join(
            f"(start={c.start}, stop={c.stop}, sharp={c.sharpness})"
            for c in self.configs
        )
        return (f"<FilterBank grid=[{self.wavelength_grid[0]}–"
                f"{self.wavelength_grid[-1]} nm], filters=[{specs}]>")

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return the filter curve for channel `idx`."""
        return self.filters[idx]

    def plot_filter(
        self,
        idx: int,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot the response of a single filter by index."""
        if ax is None:
            fig, ax = plt.subplots()
        label = f"Filter {idx}: {self.configs[idx].start}–{self.configs[idx].stop} nm"
        ax.plot(self.wavelength_grid, self.filters[idx], label=label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_title(f"Filter {idx} Response")
        ax.legend()
        return ax

    def plot_all_filters(
        self,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Overlay all filter responses."""
        if ax is None:
            fig, ax = plt.subplots()
        for i, cfg in enumerate(self.configs):
            label = f"{cfg.start}–{cfg.stop} nm"
            ax.plot(self.wavelength_grid, self.filters[i], label=label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_title("All Filter Responses")
        ax.legend()
        return ax

    def plot_filters_with_labels(
        self,
        labels: Sequence[str],
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Overlay all filter responses with custom labels.
        `labels` length must match number of filters.
        """
        if len(labels) != len(self.configs):
            raise ValueError("labels must match number of filters")
        if ax is None:
            fig, ax = plt.subplots()
        for i, label in enumerate(labels):
            ax.plot(self.wavelength_grid, self.filters[i], label=label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_title("All Filter Responses")
        ax.legend()
        return ax
