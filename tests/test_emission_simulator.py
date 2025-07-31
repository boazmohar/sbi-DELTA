import os
import numpy as np
import pytest
from sbi_delta.config import BaseConfig, ExcitationConfig, FilterConfig, PriorConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.filter_bank import FilterBank
from sbi_delta.prior_manager import PriorManager
from sbi_delta.simulator.emission_simulator import EmissionSimulator

def make_simulator(bg_dye='AF_v1', photon_budget=100):
    base_path = os.path.abspath(".")
    print("Base path for data:", base_path)
    base_cfg = BaseConfig(
        min_wavelength=400,
        max_wavelength=750,
        wavelength_step=1,
        spectra_folder=os.path.join(base_path, "data/spectra_npz"),
        dye_names=["JF479", "JF525", "JF552", "JF608", "JFX673"],
        bg_dye=bg_dye,
        photon_budget=photon_budget,
    )
    exc_cfg = ExcitationConfig(excitation_mode="min_crosstalk")
    filter_cfgs = [
        FilterConfig(start=520, stop=550, sharpness=1),
        FilterConfig(start=550, stop=580, sharpness=2),
        FilterConfig(start=580, stop=610, sharpness=2),
        FilterConfig(start=610, stop=640, sharpness=2),
        FilterConfig(start=640, stop=700, sharpness=2),
    ]
    # Only include background ratio in prior if bg_dye is set
    include_bg_ratio = bg_dye is not None
    prior_cfg = PriorConfig(
        dirichlet_concentration=5.0,
        include_background_ratio=include_bg_ratio,
        background_ratio_bounds=(0.1, 0.2)
    )
    spectra_mgr = SpectraManager(base_cfg)
    spectra_mgr.load()
    excitation_mgr = ExcitationManager(base_cfg, exc_cfg, spectra_mgr)
    filter_bank = FilterBank(base_cfg, filter_cfgs)
    prior_mgr = PriorManager(prior_cfg, base_cfg)
    sim = EmissionSimulator(
        spectra_manager=spectra_mgr,
        filter_bank=filter_bank,
        config=base_cfg,
        excitation_manager=excitation_mgr,
        prior_manager=prior_mgr,
    )
    return sim, prior_mgr

def test_simulate_no_bg_no_noise():
    sim, prior_mgr = make_simulator(bg_dye=None)
    concentrations = np.ones(5)
    counts = sim.simulate(concentrations=concentrations, add_noise=False, debug=False)
    assert counts.shape == (5, 5)
    assert np.all(counts >= 0)
    assert np.isclose(np.sum(counts), sim.config.photon_budget, atol=1e-2)

def test_simulate_with_bg_no_noise():
    sim, prior_mgr = make_simulator(bg_dye="AF_v1")
    concentrations = np.ones(6)  # 5 dyes + 1 bg
    counts = sim.simulate(concentrations=concentrations, add_noise=False, debug=False)
    assert counts.shape == (5, 5)
    assert np.all(counts >= 0)
    # Total photons should be photon_budget (signal) + photon_budget * bg_conc (background)
    bg_conc = concentrations[-1]
    expected_total = sim.config.photon_budget * (1 + bg_conc)
    assert np.isclose(np.sum(counts), expected_total, atol=1e-2)

def test_simulate_no_bg_with_noise():
    sim, prior_mgr = make_simulator(bg_dye=None)
    concentrations = np.ones(5)
    counts = sim.simulate(concentrations=concentrations, add_noise=True, debug=False)
    assert counts.shape == (5, 5)
    assert np.all(counts >= 0)
    # With noise, sum is close to photon_budget but not exact
    assert np.abs(np.sum(counts) - sim.config.photon_budget) < 5 * np.sqrt(sim.config.photon_budget)

def test_simulate_with_bg_with_noise():
    sim, prior_mgr = make_simulator(bg_dye="AF_v1")
    concentrations = np.ones(6)  # 5 dyes + 1 bg
    counts = sim.simulate(concentrations=concentrations, add_noise=True, debug=False)
    assert counts.shape == (5, 5)
    assert np.all(counts >= 0)
    bg_conc = concentrations[-1]
    expected_total = sim.config.photon_budget * (1 + bg_conc)
    assert np.abs(np.sum(counts) - expected_total) < 5 * np.sqrt(expected_total)

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
