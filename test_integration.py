#!/usr/bin/env python3
"""
Test script to demonstrate the integration of the batch_simulator functionality
with the SBISimulator class.
"""

import numpy as np
import torch
from pathlib import Path
from multiplex_sim.sbi_simulator import SBISimulator, SBIConfig, batch_simulator

def test_batch_simulator_integration():
    """Test the integration of the batch_simulator with SBISimulator."""
    
    # Setup
    fluorophore_names = ['JF525', 'JF552', 'JF608', 'JFX673', 'JF722']
    spectra_dir = Path("data/spectra_npz")
    
    # Create simulator
    config = SBIConfig(
        wavelength_range=(500, 800),
        wavelength_step=1.0,
        total_dye_photons=150.0,
        total_background_photons=20.0,
        edge_steepness=1.0
    )
    
    simulator = SBISimulator(fluorophore_names, spectra_dir, config)
    
    # Test 1: Verify SpectraManager can calculate peak wavelengths
    print("=== Test 1: Peak Wavelength Calculation ===")
    peak_wavelengths = simulator.spectra_manager.calculate_peak_emission_wavelengths(fluorophore_names)
    print("Peak wavelengths:")
    for name, wl in peak_wavelengths.items():
        print(f"  {name}: {wl} nm")
    
    # Test 2: Test batch_simulator compatibility
    print("\n=== Test 2: Batch Simulator Compatibility ===")
    
    # Create wavelength grid
    λ_grid = np.arange(500, 801, 1)
    
    # Load spectra
    interpolated_emissions = simulator.spectra_manager.load_and_interpolate_emission(fluorophore_names)
    
    # Load NADH background
    nadh_em_interp = simulator.spectra_manager.load_background_spectrum("NADH")
    
    # Create test parameters
    batch_size = 10
    n_channels = 5
    
    # Parameters: [amp_1..5, σ_1..5, center_1..5]
    theta_batch = torch.rand(batch_size, 15)
    
    # Normalize amplitudes to sum to 1
    theta_batch[:, :5] = theta_batch[:, :5] / theta_batch[:, :5].sum(dim=1, keepdim=True)
    
    # Ensure bandwidths are positive
    theta_batch[:, 5:10] = torch.abs(theta_batch[:, 5:10]) * 20 + 10  # 10-30 nm
    
    # Ensure centers are in range
    theta_batch[:, 10:15] = 500 + theta_batch[:, 10:15] * 300  # 500-800 nm
    
    # Run batch simulator
    x_batch = batch_simulator(
        theta_batch, 
        λ_grid, 
        interpolated_emissions, 
        nadh_em_interp,
        edge_steepness=1,
        total_dye_photons=150.0,
        total_background_photons=20.0,
        rng_seed=42
    )
    
    print(f"Batch simulator output shape: {x_batch.shape}")
    print(f"Sample output (first 3 samples):")
    print(x_batch[:3])
    
    # Test 3: Test new SBISimulator methods with filter parameters
    print("\n=== Test 3: New SBISimulator Methods ===")
    
    # Generate training data with filter parameters
    parameters, observations = simulator.generate_training_data_with_filters(
        n_samples=5,
        n_channels=5,
        concentration_concentration=5.0,
        center_wavelength_std=10.0
    )
    
    print(f"Parameters shape: {parameters.shape}")
    print(f"Observations shape: {observations.shape}")
    print(f"Parameter vector structure: [concentrations, center_wavelengths, bandwidths]")
    print(f"  - Concentrations: {simulator.n_fluorophores} values")
    print(f"  - Center wavelengths: 5 values")
    print(f"  - Bandwidths: 5 values")
    
    # Test 4: Test simulate_batch_with_parameters
    print("\n=== Test 4: simulate_batch_with_parameters ===")
    
    # Create test parameters
    test_params = torch.rand(3, simulator.n_fluorophores + 10)  # 5 fluorophores + 5 centers + 5 bandwidths
    
    # Normalize concentrations
    test_params[:, :simulator.n_fluorophores] = test_params[:, :simulator.n_fluorophores] / \
        test_params[:, :simulator.n_fluorophores].sum(dim=1, keepdim=True)
    
    # Ensure bandwidths are positive
    test_params[:, simulator.n_fluorophores + 5:] = torch.abs(test_params[:, simulator.n_fluorophores + 5:]) * 20 + 10
    
    # Ensure centers are in range
    test_params[:, simulator.n_fluorophores:simulator.n_fluorophores + 5] = \
        500 + test_params[:, simulator.n_fluorophores:simulator.n_fluorophores + 5] * 300
    
    # Run simulation
    sim_results = simulator.simulate_batch_with_parameters(
        test_params.numpy(),
        n_channels=5,
        add_noise=True
    )
    
    print(f"Simulation results shape: {sim_results.shape}")
    print(f"Sample results (first 3 samples):")
    print(sim_results[:3])
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    test_batch_simulator_integration()
