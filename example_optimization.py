#!/usr/bin/env python3
"""
Example script demonstrating how to use the microscope parameter optimization.

This script shows different ways to use the MicroscopeOptimizer class for
designing optimal microscope parameters.
"""

from optimize_microscope_parameters import MicroscopeOptimizer, OptimizationConfig, ExperimentalConstraints
from pathlib import Path

def example_basic_optimization():
    """Basic optimization example with default settings."""
    print("=" * 60)
    print("BASIC OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create optimizer with default settings
    optimizer = MicroscopeOptimizer()
    
    # Run optimization with automatic fluorophore selection
    report = optimizer.run_complete_optimization(
        fluorophore_names=["AF488", "AF555", "AF647"],  # Specify fluorophores
        interactive=False  # Non-interactive mode
    )
    
    return report

def example_custom_optimization():
    """Example with custom configuration."""
    print("=" * 60)
    print("CUSTOM OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create custom optimization configuration
    opt_config = OptimizationConfig(
        n_channels=4,  # Use 4 detection channels
        wavelength_bounds=(480, 720),  # Custom wavelength range
        bandwidth_bounds=(20.0, 60.0),  # Custom bandwidth range
        n_filter_trials=300,  # More optimization trials
        n_training_samples=8000,  # More training samples
        optimize_excitation=True,
        verbose=True
    )
    
    # Create experimental constraints
    constraints = ExperimentalConstraints(
        available_lasers=[405, 488, 561, 640],  # Available laser lines
        max_photon_budget=800.0,
        background_level=25.0
    )
    
    # Create optimizer with custom configuration
    optimizer = MicroscopeOptimizer(
        optimization_config=opt_config,
        experimental_constraints=constraints
    )
    
    # Run optimization with specific fluorophores
    report = optimizer.run_complete_optimization(
        fluorophore_names=["AF488", "AF532", "AF594", "AF647"],
        interactive=False
    )
    
    return report

def example_step_by_step():
    """Example showing step-by-step optimization process."""
    print("=" * 60)
    print("STEP-BY-STEP OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create optimizer
    optimizer = MicroscopeOptimizer()
    
    # Step 1: Setup simulator
    fluorophores = ["AF488", "AF555", "AF647"]
    optimizer.setup_simulator(fluorophores)
    
    # Step 2: Optimize filters only
    filter_config = optimizer.optimize_detection_filters()
    print(f"Optimal filters: {filter_config}")
    
    # Step 3: Optimize excitation wavelengths
    excitation_config = optimizer.optimize_excitation_wavelengths()
    print(f"Optimal excitation: {excitation_config}")
    
    # Step 4: Evaluate performance
    performance = optimizer.evaluate_configuration()
    print(f"Performance metrics available: {list(performance.keys())}")
    
    # Step 5: Generate visualizations
    figures = optimizer.generate_visualizations()
    print(f"Generated {len(figures)} plots")
    
    # Step 6: Generate report
    report = optimizer.generate_report()
    
    return report

if __name__ == "__main__":
    print("Microscope Parameter Optimization Examples")
    print("==========================================")
    
    # Run basic example
    try:
        print("\n1. Running basic optimization...")
        basic_report = example_basic_optimization()
        print("✓ Basic optimization completed")
    except Exception as e:
        print(f"❌ Basic optimization failed: {e}")
    
    # Run custom example
    try:
        print("\n2. Running custom optimization...")
        custom_report = example_custom_optimization()
        print("✓ Custom optimization completed")
    except Exception as e:
        print(f"❌ Custom optimization failed: {e}")
    
    # Run step-by-step example
    try:
        print("\n3. Running step-by-step optimization...")
        step_report = example_step_by_step()
        print("✓ Step-by-step optimization completed")
    except Exception as e:
        print(f"❌ Step-by-step optimization failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the 'optimization_results' directory for outputs.")
