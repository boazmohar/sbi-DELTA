#!/usr/bin/env python3
"""
Test script for prior visualization functionality.

This script demonstrates how to use the new prior visualization tools
with the enhanced SBI simulator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from multiplex_sim.sbi_simulator_with_filters import create_enhanced_sbi_simulator
from multiplex_sim.prior_visualization import PriorVisualizer, create_demo_plots


def test_prior_visualization():
    """Test the prior visualization functionality."""
    print("Testing prior visualization...")
    
    # Create simulator with background parameters
    fluorophore_names = ['AF488', 'AF555', 'AF594', 'AF647']
    simulator = create_enhanced_sbi_simulator(fluorophore_names)
    
    # Create custom prior with background
    prior_config = {
        'concentration': 2.0,
        'center_low': 500,
        'center_high': 700,
        'bandwidth_low': 15,
        'bandwidth_high': 45,
        'background_low': 5.0,
        'background_high': 150.0
    }
    
    prior = simulator.create_custom_prior(prior_config=prior_config)
    
    # Create visualizer
    visualizer = PriorVisualizer(simulator)
    visualizer.set_prior(prior)
    
    print(f"Total parameters: {prior.total_params}")
    print(f"Concentration params: {prior.n_concentration_params}")
    print(f"Center params: {prior.n_center_params}")
    print(f"Bandwidth params: {prior.n_bandwidth_params}")
    print(f"Background params: {prior.n_background_params}")
    
    # Test parameter extraction
    samples = prior.sample((10,))
    params = prior.extract_parameters(samples)
    
    print("\nParameter shapes:")
    for key, value in params.items():
        print(f"{key}: {value.shape}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: Prior distributions
    fig1 = visualizer.plot_prior_distributions(n_samples=2000)
    fig1.suptitle('Prior Distributions for Enhanced SBI Simulator', fontsize=16)
    
    # Plot 2: Correlations
    fig2 = visualizer.plot_parameter_correlations(n_samples=1000)
    fig2.suptitle('Parameter Correlations', fontsize=16)
    
    # Plot 3: Filter configuration space
    fig3 = visualizer.plot_filter_configuration_space(n_samples=1000)
    fig3.suptitle('Filter Configuration Space', fontsize=16)
    
    # Show plots
    plt.show()
    
    print("Prior visualization test completed successfully!")
    return visualizer


def test_background_integration():
    """Test background parameter integration."""
    print("\nTesting background parameter integration...")
    
    # Create simulator with background
    from multiplex_sim.sbi_simulator_with_filters import EnhancedSBIConfig
    
    config = EnhancedSBIConfig(
        include_filter_params=True,
        include_background_params=True,
        background_bounds=(10.0, 200.0)
    )
    
    fluorophore_names = ['AF488', 'AF555']
    simulator = create_enhanced_sbi_simulator(fluorophore_names, config=config)
    
    # Generate training data with background
    theta, x = simulator.generate_training_data(
        n_samples=100,
        prior_config={'background_low': 10.0, 'background_high': 200.0},
        use_custom_prior=True
    )
    
    print(f"Generated training data:")
    print(f"Parameters shape: {theta.shape}")
    print(f"Observations shape: {x.shape}")
    print(f"Parameter names: concentrations, centers, bandwidths, background")
    
    # Extract parameters including background
    from multiplex_sim.sbi_simulator_with_filters import CustomFlatPrior
    
    prior = simulator.create_custom_prior()
    params = prior.extract_parameters(theta)
    
    print(f"\nExtracted parameter shapes:")
    for key, value in params.items():
        print(f"{key}: {value.shape}")
    
    return simulator, theta, x


def test_peak_centered_distribution():
    """Test the peak-centered prior distribution for filter centers."""
    print("\nTesting peak-centered distribution...")
    
    fluorophore_names = ['AF488', 'AF555', 'AF594']
    from multiplex_sim.sbi_simulator_with_filters import EnhancedSBIConfig
    
    # Configure the simulator to use the peak-centered distribution
    config = create_enhanced_sbi_simulator(
        fluorophore_names,
        config=EnhancedSBIConfig(
            n_channels=len(fluorophore_names),
            center_wavelength_distribution='peak_centered',
            peak_centered_std=5.0  # Use a small std for a tight distribution
        )
    )
    
    # Create a custom prior with this configuration
    prior = config.create_custom_prior()
    
    # Sample from the prior
    samples = prior.sample((1000,))
    
    # Extract the center wavelengths
    params = prior.extract_parameters(samples)
    centers = params['center_wavelengths']
    
    # Get the expected peak emission wavelengths
    expected_peaks = config._get_peak_emission_wavelengths()
    
    print(f"Expected peak emission wavelengths: {expected_peaks}")
    print(f"Sampled center means: {centers.mean(dim=0).tolist()}")
    print(f"Sampled center stds: {centers.std(dim=0).tolist()}")
    
    # Check that the means of the sampled centers are close to the expected peaks
    assert torch.allclose(
        centers.mean(dim=0),
        torch.tensor(expected_peaks, dtype=torch.float32),
        atol=1.0  # Allow for some tolerance
    ), "Mean of sampled centers should be close to peak emission wavelengths"
    
    print("Peak-centered distribution test passed!")
    return config, prior


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced SBI Simulator - Prior Visualization Test")
    print("=" * 60)
    
    # Test 1: Basic visualization
    visualizer = test_prior_visualization()
    
    # Test 2: Background integration
    simulator, theta, x = test_background_integration()
    
    # Test 3: Peak-centered distribution
    test_peak_centered_distribution()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
