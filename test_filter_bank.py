# File: tests/test_filter_bank.py
"""
Unit tests for sbi_delta.filter_bank.FilterBank using FilterConfig(start, stop, sharpness).
"""
import numpy as np
import pytest

from sbi_delta.config import BaseConfig, FilterConfig
from sbi_delta.filter_bank import FilterBank


def test_empty_bank():
    """Zero filters should produce an empty (0 × n_wavelengths) array."""
    base_cfg = BaseConfig(min_wavelength=300.0, max_wavelength=800.0, wavelength_step=10.0)
    bank = FilterBank(base_cfg, [])
    filt = bank.filters
    assert filt.shape == (0, bank.wavelength_grid.size)


def test_single_filter_shape_and_bounds():
    """A single filter covers the grid, values in [0,1], smooth edges."""
    base_cfg = BaseConfig(min_wavelength=350.0, max_wavelength=550.0, wavelength_step=5.0)
    cfg = FilterConfig(start=400.0, stop=500.0, sharpness=10.0)
    bank = FilterBank(base_cfg, [cfg])

    grid = bank.wavelength_grid
    arr = bank.filters[0]

    # one channel, correct length
    assert bank.filters.shape == (1, grid.size)
    # Should be between 0 and 1
    assert np.all(arr >= 0) and np.all(arr <= 1)
    # At midpoint (450), it should be closer to 1 than 0
    mid_idx = np.argmin(np.abs(grid - 450.0))
    assert arr[mid_idx] > 0.5


def test_top_hat_limit_with_small_sharpness():
    """
    Very small sharpness approximates a top-hat: inside passband near 1,
    edges at start/stop near 0, with inflection ~0.5.
    """
    base_cfg = BaseConfig(min_wavelength=0.0, max_wavelength=300.0, wavelength_step=50.0)
    cfg = FilterConfig(start=100.0, stop=200.0, sharpness=0.01)
    bank = FilterBank(base_cfg, [cfg])

    grid = bank.wavelength_grid
    arr = bank.filters[0]

    # At grid edges outside [100,200]
    assert pytest.approx(0.0, abs=1e-2) == arr[0]  # 0 nm
    # Find indices for 100,150,200
    idx100 = list(grid).index(100.0)
    idx150 = list(grid).index(150.0)
    idx200 = list(grid).index(200.0)
    assert pytest.approx(0.5, rel=1e-2) == arr[idx100]
    assert pytest.approx(1.0, rel=1e-2) == arr[idx150]
    assert pytest.approx(0.5, rel=1e-2) == arr[idx200]
    # Beyond stop
    assert pytest.approx(0.0, abs=1e-2) == arr[-1]


def test_multiple_filters_sorting_and_no_overlap():
    """Configs out of order get sorted by start, and overlapping configs raise."""
    base_cfg = BaseConfig(min_wavelength=300.0, max_wavelength=900.0, wavelength_step=50.0)
    # out-of-order definitions
    cfg2 = FilterConfig(start=600.0, stop=650.0, sharpness=5.0)
    cfg1 = FilterConfig(start=400.0, stop=450.0, sharpness=5.0)
    bank = FilterBank(base_cfg, [cfg2, cfg1])
    # Internally sorted by start
    starts = [c.start for c in bank.configs]
    assert starts == [400.0, 600.0]

    # Overlap: stop1 > start2
    bad1 = FilterConfig(start=400.0, stop=500.0, sharpness=5.0)
    bad2 = FilterConfig(start=450.0, stop=550.0, sharpness=5.0)
    with pytest.raises(ValueError):
        FilterBank(base_cfg, [bad1, bad2])


def test_repr_lists_configs():
    """__repr__ should list each filter's parameters in order."""
    base_cfg = BaseConfig(min_wavelength=300.0, max_wavelength=900.0, wavelength_step=100.0)
    cfg1 = FilterConfig(start=400.0, stop=450.0, sharpness=2.0)
    cfg2 = FilterConfig(start=500.0, stop=550.0, sharpness=3.0)
    bank = FilterBank(base_cfg, [cfg2, cfg1])
    rep = repr(bank)
    # Should mention both filters in sorted order
    assert "start=400.0" in rep and "stop=450.0" in rep and "sharp=2.0" in rep
    assert "start=500.0" in rep and "stop=550.0" in rep and "sharp=3.0" in rep
