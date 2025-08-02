import os
import numpy as np
import pytest
import torch
from sbi_delta.config import BaseConfig, ExcitationConfig, PriorConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.prior_manager import PriorManager
from sbi_delta.simulator.filter_prior_simulator import FilterPriorSimulator
from sbi_delta.filter_trainer import FilterTrainer

def make_test_setup(n_filters=5):
    """Create a test setup with minimal configuration."""
    base_cfg = BaseConfig(
        min_wavelength=400,
        max_wavelength=750,
        wavelength_step=1,
        spectra_folder=os.path.join(os.path.abspath("."), "data/spectra_npz"),
        dye_names=["JF479", "JF525", "JF552"],  # Using fewer dyes for faster testing
        photon_budget=100,
    )
    
    exc_cfg = ExcitationConfig(
        excitation_mode="min_crosstalk"
    )
    
    prior_cfg = PriorConfig(
        dirichlet_concentration=5.0,
        include_filter_params=True,
        n_filters=n_filters,
        max_filter_width=50.0,
        min_filter_width=10.0
    )
    
    spectra_mgr = SpectraManager(base_cfg)
    spectra_mgr.load()
    excitation_mgr = ExcitationManager(base_cfg, exc_cfg, spectra_mgr)
    prior_mgr = PriorManager(prior_cfg, base_cfg, excitation_mgr)
    
    sim = FilterPriorSimulator(
        spectra_manager=spectra_mgr,
        config=base_cfg,
        excitation_manager=excitation_mgr,
        prior_manager=prior_mgr,
        n_filters=n_filters,
        max_filter_width=prior_cfg.max_filter_width
    )
    
    return sim, prior_mgr

def test_trainer_initialization():
    """Test that trainer initializes correctly."""
    sim, prior_mgr = make_test_setup()
    
    trainer = FilterTrainer(
        simulator=sim,
        prior_manager=prior_mgr,
        training_batch_size=50,
        num_workers=1,
        save_dir="test_results"
    )
    
    assert trainer.simulator == sim
    assert trainer.prior == prior_mgr.get_joint_prior()
    assert trainer.training_batch_size == 50
    assert trainer._theta is None
    assert trainer._x is None

def test_simulation_generation():
    """Test that training data generation works."""
    sim, prior_mgr = make_test_setup()
    
    trainer = FilterTrainer(
        simulator=sim,
        prior_manager=prior_mgr,
        training_batch_size=50,
        num_workers=1
    )
    
    n_sim = 10  # Small number for testing
    trainer.simulate_for_sbi(n_sim)
    
    # Check shapes
    assert trainer._theta.shape == (n_sim, 11)  # 3 dyes + (2 params × 4 filters)
    assert trainer._x.shape == (n_sim, 15)  # 3 exc × 5 filters = 15 measurements
    
    # Check data types
    assert isinstance(trainer._theta, torch.Tensor)
    assert isinstance(trainer._x, torch.Tensor)
    
    # Check values
    assert torch.all(trainer._x >= 0)  # Photon counts should be non-negative

def test_small_training_run():
    """Test a small complete training run."""
    sim, prior_mgr = make_test_setup(n_filters=3)  # Fewer filters for faster testing
    
    trainer = FilterTrainer(
        simulator=sim,
        prior_manager=prior_mgr,
        training_batch_size=50,
        num_workers=1
    )
    
    # Generate small training dataset
    n_sim = 100
    trainer.simulate_for_sbi(n_sim)
    
    # Train for a few steps
    density_estimator = trainer.train_density_estimator()
    assert density_estimator is not None
    
    # Test posterior building
    observation = sim.simulate(prior_mgr.get_joint_prior().sample((1,)).numpy()[0])
    posterior = trainer.build_posterior()
    assert posterior is not None
    
    # Sample from posterior
    n_samples = 10
    samples = posterior.sample((n_samples,))
    assert samples.shape == (n_samples, 9)  # 3 dyes + (2 params × 3 filters)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])