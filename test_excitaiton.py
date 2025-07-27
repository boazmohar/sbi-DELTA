import numpy as np
import pytest
from sbi_delta.config import BaseConfig, ExcitationConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager

class DummySpectraManager:
    """A dummy spectra manager for testing excitation logic, including background dye."""
    def __init__(self, dye_names, peaks=None, bg_dye=None, bg_peak=None):
        self.dye_names = dye_names
        self.wavelength_grid = np.arange(400, 701, 1)
        self._excitation = []
        self.excitation_names = list(dye_names)
        for i, dye in enumerate(dye_names):
            arr = np.zeros_like(self.wavelength_grid, dtype=float)
            peak = peaks[i] if peaks is not None else 500 + 10*i
            arr[np.abs(self.wavelength_grid - peak).argmin()] = 1.0
            self._excitation.append(arr)
        self._excitation = np.array(self._excitation)
        # Add background dye if provided
        self.bg_dye = bg_dye
        if bg_dye is not None and bg_peak is not None:
            self.excitation_names.append(bg_dye)
            arr = np.zeros_like(self.wavelength_grid, dtype=float)
            arr[np.abs(self.wavelength_grid - bg_peak).argmin()] = 1.0
            self._excitation = np.vstack([self._excitation, arr])
    def get_excitation(self, name):
        idx = self.excitation_names.index(name)
        return self._excitation[idx]

def test_manual_mode():
    dyes = ["A", "B"]
    manual_wl = [450, 550]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="manual", manual_wavelengths=manual_wl)
    mgr = DummySpectraManager(dyes)
    emgr = ExcitationManager(cfg, excfg, mgr)
    np.testing.assert_array_equal(emgr.get_wavelengths(), manual_wl)

def test_peak_mode():
    dyes = ["A", "B"]
    peaks = [480, 600]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="peak")
    mgr = DummySpectraManager(dyes, peaks=peaks)
    emgr = ExcitationManager(cfg, excfg, mgr)
    np.testing.assert_array_equal(emgr.get_wavelengths(), peaks)

def test_manual_mode_wrong_length():
    dyes = ["A", "B", "C"]
    manual_wl = [450, 550]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="manual", manual_wavelengths=manual_wl)
    mgr = DummySpectraManager(dyes)
    with pytest.raises(ValueError):
        ExcitationManager(cfg, excfg, mgr)

def test_min_crosstalk_mode_simple():
    dyes = ["A", "B"]
    peaks = [480, 600]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="min_crosstalk", search_range=(470, 610))
    mgr = DummySpectraManager(dyes, peaks=peaks)
    emgr = ExcitationManager(cfg, excfg, mgr)
    np.testing.assert_array_equal(emgr.get_wavelengths(), peaks)

def test_min_crosstalk_mode_overlap():
    dyes = ["A", "B"]
    peaks = [500, 500]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="min_crosstalk", search_range=(495, 505))
    mgr = DummySpectraManager(dyes, peaks=peaks)
    emgr = ExcitationManager(cfg, excfg, mgr)
    np.testing.assert_array_equal(emgr.get_wavelengths(), [500, 500])

def test_min_crosstalk_with_bg():
    dyes = ["A", "B"]
    peaks = [480, 600]
    bg_dye = "BG"
    bg_peak = 550
    cfg = BaseConfig(dye_names=dyes, bg_dye=bg_dye)
    excfg = ExcitationConfig(excitation_mode="min_crosstalk", search_range=(470, 610))
    mgr = DummySpectraManager(dyes, peaks=peaks, bg_dye=bg_dye, bg_peak=bg_peak)
    emgr = ExcitationManager(cfg, excfg, mgr)
    # The background dye should be included in the crosstalk calculation, but not optimized
    # The optimal wavelengths for A and B should still be their peaks
    np.testing.assert_array_equal(emgr.get_wavelengths(), peaks)
    # Check that the crosstalk matrix includes the background dye
    ax = emgr.plot_crosstalk_matrix(include_bg=True)
    assert len(ax.get_xticklabels()) == 3  # A, B, BG
    assert len(ax.get_yticklabels()) == 3

def test_invalid_mode():
    dyes = ["A"]
    cfg = BaseConfig(dye_names=dyes)
    excfg = ExcitationConfig(excitation_mode="not_a_mode")
    mgr = DummySpectraManager(dyes)
    with pytest.raises(ValueError):
        ExcitationManager(cfg, excfg, mgr)