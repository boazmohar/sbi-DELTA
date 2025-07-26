"""Tests for the configuration classes in :mod:`sbi_delta.config`.

Each test logs its progress using the standard :mod:`logging` module.  The
logging configuration is set at module import time to ensure that log
messages appear when running ``pytest -s``.  Individual tests then emit
informational messages describing what is being tested.  Where exceptions
are expected, the tests assert that the correct type of error is raised.
"""

import logging
import pytest

from sbi_delta.config import (
    BaseConfig,
    FilterConfig,
    ExcitationConfig,
)


# Configure logging for the test module.  By default pytest captures log output,
# but it can be displayed with ``pytest -s`` or ``-o log_cli=true``.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_baseconfig_defaults():
    cfg = BaseConfig()
    assert cfg.min_wavelength == 350.0
    assert cfg.max_wavelength == 800.0
    assert cfg.wavelength_step == 1.0
    assert cfg.photon_budget == 1e5
    assert cfg.spectra_folder == ""
    assert cfg.dye_names == []
    assert cfg.bg_dye is None

def test_baseconfig_custom_fields():
    cfg = BaseConfig(
        min_wavelength=400,
        max_wavelength=700,
        wavelength_step=2,
        photon_budget=12345,
        spectra_folder="myfolder",
        dye_names=["AF488", "AF647"],
        bg_dye="AF488"
    )
    assert cfg.spectra_folder == "myfolder"
    assert cfg.dye_names == ["AF488", "AF647"]
    assert cfg.bg_dye == "AF488"

def test_baseconfig_invalid_range():
    with pytest.raises(ValueError):
        BaseConfig(min_wavelength=800, max_wavelength=700)

def test_baseconfig_invalid_step():
    with pytest.raises(ValueError):
        BaseConfig(wavelength_step=0)

def test_baseconfig_invalid_photon_budget():
    with pytest.raises(ValueError):
        BaseConfig(photon_budget=0)

def test_baseconfig_bg_dye_not_in_dye_names():
    # Should not raise, but warn (cannot test warning easily here)
    cfg = BaseConfig(
        spectra_folder="folder",
        dye_names=["AF488"],
        bg_dye="AF555"
    )
    assert cfg.bg_dye == "AF555"
    assert "AF555" not in cfg.dye_names

def test_filter_config_defaults() -> None:
    """Default FilterConfig values are correctly set."""
    logger.info("Starting test_filter_config_defaults")
    fc = FilterConfig()
    assert fc.start == 400.0
    assert fc.stop == 700.0
    assert fc.sharpness == 1.0
    logger.info("Completed test_filter_config_defaults")


def test_filter_config_invalid_range() -> None:
    """start >= stop should raise ValueError."""
    logger.info("Starting test_filter_config_invalid_range")
    with pytest.raises(ValueError):
        FilterConfig(start=700.0, stop=400.0)
    with pytest.raises(ValueError):
        FilterConfig(start=500.0, stop=500.0)
    logger.info("Completed test_filter_config_invalid_range")


def test_filter_config_invalid_sharpness() -> None:
    """Non-positive sharpness should raise ValueError."""
    logger.info("Starting test_filter_config_invalid_sharpness")
    with pytest.raises(ValueError):
        FilterConfig(sharpness=0.0)
    with pytest.raises(ValueError):
        FilterConfig(sharpness=-5.0)
    logger.info("Completed test_filter_config_invalid_sharpness")


def test_filter_config_custom_values() -> None:
    """Custom start, stop, and sharpness values are respected."""
    logger.info("Starting test_filter_config_custom_values")
    fc = FilterConfig(start=450.0, stop=650.0, sharpness=20.0)
    assert fc.start == 450.0
    assert fc.stop == 650.0
    assert fc.sharpness == 20.0
    logger.info("Completed test_filter_config_custom_values")


def test_excitation_config_manual() -> None:
    """Manual excitation wavelengths are accepted when enabled."""
    logger.info("Starting test_excitation_config_manual")
    ex = ExcitationConfig(use_manual_wavelengths=True, manual_wavelengths=[405.0, 488.0])
    assert ex.use_manual_wavelengths is True
    assert ex.manual_wavelengths == [405.0, 488.0]
    logger.info("Completed test_excitation_config_manual")


def test_excitation_config_missing_manual() -> None:
    """Missing manual wavelengths should raise ValueError when enabled."""
    logger.info("Starting test_excitation_config_missing_manual")
    with pytest.raises(ValueError):
        ExcitationConfig(use_manual_wavelengths=True)
    logger.info("Completed test_excitation_config_missing_manual")


def test_excitation_config_invalid_search_range() -> None:
    """Invalid search ranges should raise ValueError."""
    logger.info("Starting test_excitation_config_invalid_search_range")
    with pytest.raises(ValueError):
        ExcitationConfig(search_range=(600.0, 500.0))
    logger.info("Completed test_excitation_config_invalid_search_range")
