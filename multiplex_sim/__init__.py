"""
Multiplex Simulation Package

A Python package for simulating multiplexed fluorescence microscopy experiments.
This package provides tools for:
- Simulating fluorescence emission and detection
- Optimizing excitation wavelengths
- Processing spectral data
- Visualizing results
"""

from .Microscope import (
    MicroscopeConfig,
    gaussian_emission,
    create_channel_filters,
    simulate_detected_signal,
    simulate_photon_detection,
    load_excitation_spectra,
    calculate_crosstalk_matrix,
    excitation_cost,
    find_optimal_excitation,
    generate_dye_combinations
)

from .io import (
    process_csv,
    process_xlsx,
    process_txt_pair,
    save_npz,
    process_spectra_folder,
    list_fluorophores
)

from .plotting import (
    plot_fluorophores,
    plot_crosstalk_matrix,
    plot_detection_channels,
    plot_simulation_results
)

from .sbi_simulator import (
    SBIConfig,
    SBISimulator,
    FilterBank,
    SpectraManager,
    create_sbi_simulator
)

from .sbi_trainer import (
    FlatPrior,
    TrainingConfig,
    ExperimentConfig,
    SBITrainer,
    run_sbi_experiment
)

from .excitation_analysis import (
    plot_excitation_crosstalk_matrix,
    analyze_excitation_crosstalk,
    plot_excitation_spectra_with_lasers,
    compare_excitation_strategies,
    plot_background_excitation_analysis,
    generate_excitation_optimization_report
)

from .advanced_optimization import (
    OptimizationConfig,
    AdvancedExcitationOptimizer,
    find_optimal_excitation_advanced
)
