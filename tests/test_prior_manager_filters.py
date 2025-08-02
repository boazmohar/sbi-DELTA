"""Test PriorManager with stick breaking filter prior."""

import os
import sys
import numpy as np
import pytest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbi_delta.config import BaseConfig, ExcitationConfig, PriorConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.prior_manager import PriorManager

def make_test_setup():
    """Create a test setup with minimal configuration."""
    base_cfg = BaseConfig(
        min_wavelength=400,
        max_wavelength=700,  # 300nm range
        wavelength_step=1,
        spectra_folder=os.path.join(os.path.abspath("."), "data/spectra_npz"),
        dye_names=["JF479", "JF525", "JF552"],  # Using fewer dyes for testing
        photon_budget=100,
    )
    
    exc_cfg = ExcitationConfig(
        excitation_mode="min_crosstalk"
    )
    
    prior_cfg = PriorConfig(
        dirichlet_concentration=5.0,
        include_filter_params=True,  # Enable filter optimization
        n_filters=3,
        max_filter_width=50.0,
        min_filter_width=10.0
    )
    
    spectra_mgr = SpectraManager(base_cfg)
    spectra_mgr.load()
    excitation_mgr = ExcitationManager(base_cfg, exc_cfg, spectra_mgr)
    prior_mgr = PriorManager(prior_cfg, base_cfg, excitation_mgr)
    
    return prior_mgr, base_cfg

def test_prior_manager_initialization():
    """Test that prior manager initializes with filter prior."""
    prior_mgr, _ = make_test_setup()
    
    # Check that filter prior is created
    filter_prior = prior_mgr.get_filter_prior()
    assert filter_prior is not None

def test_joint_prior_sampling():
    """Test sampling from joint prior including filter parameters."""
    prior_mgr, base_cfg = make_test_setup()
    
    # Get joint prior
    joint_prior = prior_mgr.get_joint_prior()
    
    # Sample from joint prior
    n_samples = 100
    samples = joint_prior.sample((n_samples,))
    
    # Check sample dimensions
    n_dyes = len(base_cfg.dye_names)
    n_segments = 2 * prior_mgr.config.n_filters - 1  # filters + gaps
    expected_dim = n_dyes + n_segments + 1  # dyes + segments + start_type
    assert samples.shape == (n_samples, expected_dim)
    
    # Check concentrations sum to 1
    concentrations = samples[:, :n_dyes]
    assert torch.allclose(concentrations.sum(dim=1), torch.ones(n_samples))
    
    # Get filter parameters
    start_type = samples[:, n_dyes]
    filter_params = samples[:, n_dyes+1:]
    
    # Check start type is binary
    assert torch.all((start_type == 0) | (start_type == 1))
    
    # Check filter parameters
    total_width = base_cfg.max_wavelength - base_cfg.min_wavelength
    assert torch.all(filter_params >= prior_mgr.config.min_filter_width)
    assert torch.all(filter_params <= prior_mgr.config.max_filter_width)
    assert torch.all(filter_params.sum(dim=1) <= total_width)

def test_filter_configurations():
    """Test different filter configurations from prior."""
    prior_mgr, base_cfg = make_test_setup()
    joint_prior = prior_mgr.get_joint_prior()
    
    # Sample multiple configurations
    n_samples = 1000
    samples = joint_prior.sample((n_samples,))
    
    # Check we get both filter-first and gap-first configurations
    start_types = samples[:, len(base_cfg.dye_names)]
    n_filter_first = (start_types == 1).sum()
    n_gap_first = (start_types == 0).sum()
    
    # With enough samples, should see both types
    assert n_filter_first > 0
    assert n_gap_first > 0
    
    # Probability should be roughly equal (within reasonable bounds)
    ratio = n_filter_first / n_samples
    assert 0.4 <= ratio <= 0.6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])