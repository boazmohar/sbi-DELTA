import os
import numpy as np
import pytest
from sbi_delta.config import BaseConfig, ExcitationConfig, PriorConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.prior_manager import PriorManager
from sbi_delta.simulator.filter_prior_simulator import FilterPriorSimulator

def make_simulator(bg_dye=None, n_filters=5, max_filter_width=50.0):
    base_path = os.path.abspath(".")
    base_cfg = BaseConfig(
        min_wavelength=400,
        max_wavelength=750,
        wavelength_step=1,
        spectra_folder=os.path.join(base_path, "data/spectra_npz"),
        dye_names=["JF479", "JF525", "JF552", "JF608", "JFX673"],
        bg_dye=bg_dye,
        photon_budget=100,
    )
    
    exc_cfg = ExcitationConfig(
        excitation_mode="min_crosstalk"
    )
    
    prior_cfg = PriorConfig(
        dirichlet_concentration=5.0,
        include_background_ratio=bg_dye is not None,
        background_ratio_bounds=(0.1, 0.2),
        include_filter_params=True,
        n_filters=n_filters,
        max_filter_width=max_filter_width
    )
    
    spectra_mgr = SpectraManager(base_cfg)
    spectra_mgr.load()
    excitation_mgr = ExcitationManager(base_cfg, exc_cfg, spectra_mgr)
    prior_mgr = PriorManager(prior_cfg, base_cfg)
    
    sim = FilterPriorSimulator(
        spectra_manager=spectra_mgr,
        config=base_cfg,
        excitation_manager=excitation_mgr,
        prior_manager=prior_mgr,
        n_filters=n_filters,
        max_filter_width=max_filter_width
    )
    return sim, prior_mgr

def test_filter_prior_simulator_no_bg():
    sim, prior_mgr = make_simulator(bg_dye=None)
    
    # Sample parameters from prior
    params = prior_mgr.get_joint_prior().sample()
    
    # Run simulation
    counts = sim.simulate(params=params, add_noise=False)
    
    # Check output shape and constraints
    assert counts.shape == (5, 5)  # n_excitation x n_filters
    assert np.all(counts >= 0)  # Non-negative counts
    assert np.isclose(np.sum(counts), sim.config.photon_budget, atol=1e-2)  # Total photon budget
    
    # Check filter parameters
    filter_params = params[5:].numpy()  # Skip concentration parameters
    for i in range(sim.n_filters):
        start = filter_params[i*2]
        width = filter_params[i*2 + 1]
        stop = start + width
        
        # Check constraints
        assert start >= sim.excitation_manager.get_wavelengths()[i]  # Start after excitation
        assert width <= sim.max_filter_width  # Width within limit
        assert stop <= sim.config.max_wavelength  # Stop within range

def test_filter_prior_simulator_with_bg():
    sim, prior_mgr = make_simulator(bg_dye="AF_v1")
    
    # Sample parameters from prior
    params = prior_mgr.get_joint_prior().sample()
    
    # Run simulation
    counts = sim.simulate(params=params, add_noise=False)
    
    # Check output shape and constraints
    assert counts.shape == (5, 5)  # n_excitation x n_filters
    assert np.all(counts >= 0)  # Non-negative counts
    
    # Extract parameters
    n_dyes = len(sim.config.dye_names)
    concentrations = params[:n_dyes].numpy()
    bg_ratio = params[n_dyes].item()
    
    # Check total matches photon budget with background
    expected_total = sim.config.photon_budget * (1 + bg_ratio)
    assert np.isclose(np.sum(counts), expected_total, atol=1e-2)