#!/usr/bin/env python3
"""
Test script for the enhanced SBI simulator.
"""

import numpy as np
import torch
import sys
import os

# Add the current directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_sim.sbi_simulator_with_filters import (
    EnhancedSBISimulator, 
    EnhancedSBIConfig
)

def test_enhanced_simulator():
    """Test the enhanced simulator functionality."""
    
    print("Testing Enhanced SBI Simulator...")
    
    # Test 1: Basic initialization
    print("\n1. Testing initialization...")
    fluorophore_names = ["AF488", "AF555"]
    config = EnhancedSBIConfig(n_channels=2, include_filter_params=True)
    simulator = EnhancedSBISimulator(fluorophore_names, "data/spectra_npz", config)

    assert simulator.total_params == 2 + 2*2  # 2 concentrations + 2 centers + 2 bandwidths
    print("✓ Initialization successful")
    
    # Test 2: Parameter extraction
    print("\n2. Testing parameter extraction...")
    theta = np.array([0.5, 0.3, 550.0, 600.0, 25.0, 30.0])
    params = simulator.extract_parameters(theta)
    
    assert params['concentrations'].shape == (1, 2)
    assert params['center_wavelengths'].shape == (1, 2)
    assert params['bandwidths'].shape == (1, 2)
    print("✓ Parameter extraction successful")
    
    # Test 3: Simulation
    print("\n3. Testing simulation...")
    theta_batch = np.array([
        [0.5, 0.3, 550.0, 600.0, 25.0, 30.0],
        [0.2, 0.7, 540.0, 590.0, 20.0, 35.0]
    ])
    
    x = simulator.simulate_batch(theta_batch, add_noise=False)
    assert x.shape == (2, 2)
    assert torch.all(x >= 0)
    print("✓ Simulation successful")
    
    # Test 4: Training data generation
    print("\n4. Testing training data generation...")
    theta_train, x_train = simulator.generate_training_data(
        n_samples=10,
        prior_config={
            'concentration_prior_type': 'dirichlet',
            'concentration_params': {'concentration': 1.0}
        },
        use_custom_prior=True
    )
    assert theta_train.shape == (10, 6)  # 2 concentrations + 2 centers + 2 bandwidths
    assert x_train.shape == (10, 2)
    print("✓ Training data generation successful")
    
    # Test 5: Without filter parameters
    print("\n5. Testing without filter parameters...")
    config_no_filters = EnhancedSBIConfig(n_channels=2, include_filter_params=False)
    simulator_no_filters = EnhancedSBISimulator(fluorophore_names, "data/spectra_npz", config_no_filters)
    
    theta_no_filters = np.array([[0.5, 0.3], [0.2, 0.7]])
    x_no_filters = simulator_no_filters.simulate_batch(theta_no_filters)
    assert x_no_filters.shape == (2, 2)
    print("✓ No-filter simulation successful")

    # Test 6: Initialization with background parameters
    print("\n6. Testing initialization with background parameters...")
    config_with_bg = EnhancedSBIConfig(n_channels=2, include_filter_params=True, include_background_params=True)
    simulator_with_bg = EnhancedSBISimulator(fluorophore_names, "data/spectra_npz", config_with_bg)
    
    assert simulator_with_bg.total_params == 2 + 2*2 + 1  # 2 concentrations + 2 centers + 2 bandwidths + 1 background
    assert hasattr(simulator_with_bg, 'background_slice')
    print("✓ Initialization with background successful")

    # Test 7: Parameter extraction with background
    print("\n7. Testing parameter extraction with background...")
    theta_with_bg = np.array([0.5, 0.3, 550.0, 600.0, 25.0, 30.0, 100.0])
    params_with_bg = simulator_with_bg.extract_parameters(theta_with_bg)
    
    assert params_with_bg['concentrations'].shape == (1, 2)
    assert params_with_bg['center_wavelengths'].shape == (1, 2)
    assert params_with_bg['bandwidths'].shape == (1, 2)
    assert params_with_bg['background'].shape == (1, 1)
    assert np.allclose(params_with_bg['background'], 100.0)
    print("✓ Parameter extraction with background successful")

    # Test 8: Simulation with background
    print("\n8. Testing simulation with background...")
    theta_low_bg = np.array([[0.5, 0.5, 550.0, 600.0, 25.0, 30.0, 10.0]])
    theta_high_bg = np.array([[0.5, 0.5, 550.0, 600.0, 25.0, 30.0, 200.0]])
    
    x_low_bg = simulator_with_bg.simulate_batch(theta_low_bg, add_noise=False)
    x_high_bg = simulator_with_bg.simulate_batch(theta_high_bg, add_noise=False)
    
    assert x_high_bg.sum() > x_low_bg.sum()
    print("✓ Simulation with background successful")

    # Test 9: Training data generation with background
    print("\n9. Testing training data generation with background...")
    theta_train_bg, x_train_bg = simulator_with_bg.generate_training_data(
        n_samples=10,
        prior_config={},
        use_custom_prior=True
    )
    
    assert theta_train_bg.shape == (10, 7)  # 2 concentrations + 2 centers + 2 bandwidths + 1 background
    assert x_train_bg.shape == (10, 2)
    print("✓ Training data generation with background successful")

    # Test 10: Custom prior with background
    print("\n10. Testing custom prior with background...")
    prior = simulator_with_bg.create_custom_prior()
    assert prior.include_background
    
    sample = prior.sample((5,))
    assert sample.shape == (5, 7)
    print("✓ Custom prior with background successful")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_enhanced_simulator()
