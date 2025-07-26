# File: tests/test_spectra_manager.py
"""
Unit tests for SpectraManager with fixed NPZ keys:
'wavelengths_emission', 'emission', 'wavelengths_excitation', 'excitation'.
"""
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
import logging

from sbi_delta.config import BaseConfig
from sbi_delta.spectra_manager import SpectraManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_npz(path, wl_em, em, wl_ex, ex):
    np.savez(path,
             wavelengths_emission=wl_em,
             emission=em,
             wavelengths_excitation=wl_ex,
             excitation=ex)


def test_list_npz_and_invalid(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    # Create dummy npz files
    for dye in ["A", "B"]:
        (data / f"{dye}.npz").write_bytes(b"")
    files = SpectraManager.list_npz(data)
    assert len(files) == 2
    with pytest.raises(ValueError):
        SpectraManager.list_npz(data / "nonexistent")


def test_load_and_access(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    # Create AF488 npz
    wl_em = np.array([400.0, 500.0])
    em = np.array([0.0, 2.0])
    wl_ex = np.array([100.0, 200.0])
    ex = np.array([1.0, 0.0])
    create_npz(data / "AF488.npz", wl_em, em, wl_ex, ex)

    cfg = BaseConfig(min_wavelength=0.0, max_wavelength=600.0, wavelength_step=100.0)
    mgr = SpectraManager(cfg, data, ["AF488", "AF647"])
    mgr.load()

        # repr should list only the *loaded* dyes in emission/excitation
    rep = repr(mgr)
    print(rep)
    assert "Emission loaded: ['AF488']" in rep
    assert "Excitation loaded: ['AF488']" in rep

    # Full matrices shape
    emi = mgr.get_emission()
    exc = mgr.get_excitation()
    assert emi.shape == (1, len(mgr.wavelength_grid))
    assert exc.shape == (1, len(mgr.wavelength_grid))

    # Single dye spectrum
    idx_em = list(mgr.wavelength_grid).index(500.0)
    assert np.isclose(mgr.get_emission("AF488")[idx_em], 1.0)
    idx_ex = list(mgr.wavelength_grid).index(100.0)
    assert np.isclose(mgr.get_excitation("AF488")[idx_ex], 1.0)

    # Unknown dye should KeyError
    with pytest.raises(KeyError):
        mgr.get_emission("AF647")
    with pytest.raises(KeyError):
        mgr.get_excitation("AF647")


def test_missing_file_skip(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cfg = BaseConfig(min_wavelength=400.0, max_wavelength=500.0, wavelength_step=50.0)
    mgr = SpectraManager(cfg, data, ["NONE"])
    mgr.load()
    # Empty matrices
    assert mgr.get_emission().shape == (0, len(mgr.wavelength_grid))
    assert mgr.get_excitation().shape == (0, len(mgr.wavelength_grid))
    # Accessing specific dye errors
    with pytest.raises(KeyError):
        mgr.get_emission("NONE")
    with pytest.raises(KeyError):
        mgr.get_excitation("NONE")
