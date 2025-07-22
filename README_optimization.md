# Microscope Parameter Optimization

This directory contains a complete script for designing optimal microscope parameters using the SBI simulator's `optimize_filter_configuration` function and related tools from the multiplex_sim package.

## Overview

The optimization script provides a comprehensive solution for:
- **Filter Configuration Optimization**: Uses the `optimize_filter_configuration` method from `SBISimulator`
- **Excitation Wavelength Optimization**: Uses existing optimization functions from `Microscope.py`
- **Performance Evaluation**: Uses SBI training to assess multiplexing performance
- **Comprehensive Reporting**: Generates detailed analysis with visualizations

## Files

- `optimize_microscope_parameters.py`: Main optimization script with complete functionality
- `example_optimization.py`: Example usage scripts showing different optimization approaches
- `README_optimization.md`: This documentation file

## Quick Start

### Basic Usage

```bash
# Interactive mode - select fluorophores interactively
python optimize_microscope_parameters.py

# Specify fluorophores directly
python optimize_microscope_parameters.py --fluorophores AF488 AF555 AF647

# Automatic fluorophore selection
python optimize_microscope_parameters.py --auto-select
```

### Advanced Usage

```bash
# Custom optimization with 4 channels and more trials
python optimize_microscope_parameters.py \
    --fluorophores AF488 AF532 AF594 AF647 \
    --n-channels 4 \
    --n-filter-trials 500 \
    --n-training-samples 10000

# Skip excitation optimization
python optimize_microscope_parameters.py \
    --fluorophores AF488 AF555 AF647 \
    --no-excitation-opt

# Quiet mode without plots
python optimize_microscope_parameters.py \
    --fluorophores AF488 AF555 AF647 \
    --quiet --no-plots
```

## Python API Usage

### Basic Optimization

```python
from optimize_microscope_parameters import MicroscopeOptimizer

# Create optimizer with default settings
optimizer = MicroscopeOptimizer()

# Run complete optimization
report = optimizer.run_complete_optimization(
    fluorophore_names=["AF488", "AF555", "AF647"],
    interactive=False
)
```

### Custom Configuration

```python
from optimize_microscope_parameters import (
    MicroscopeOptimizer, OptimizationConfig, ExperimentalConstraints
)

# Custom optimization configuration
opt_config = OptimizationConfig(
    n_channels=4,
    wavelength_bounds=(480, 720),
    bandwidth_bounds=(20.0, 60.0),
    n_filter_trials=300,
    n_training_samples=8000,
    optimize_excitation=True,
    verbose=True
)

# Experimental constraints
constraints = ExperimentalConstraints(
    available_lasers=[405, 488, 561, 640],
    max_photon_budget=800.0,
    background_level=25.0
)

# Create optimizer
optimizer = MicroscopeOptimizer(
    optimization_config=opt_config,
    experimental_constraints=constraints
)

# Run optimization
report = optimizer.run_complete_optimization(
    fluorophore_names=["AF488", "AF532", "AF594", "AF647"],
    interactive=False
)
```

### Step-by-Step Optimization

```python
# Create optimizer
optimizer = MicroscopeOptimizer()

# Step 1: Setup simulator
optimizer.setup_simulator(["AF488", "AF555", "AF647"])

# Step 2: Optimize detection filters
filter_config = optimizer.optimize_detection_filters()

# Step 3: Optimize excitation wavelengths
excitation_config = optimizer.optimize_excitation_wavelengths()

# Step 4: Evaluate performance
performance = optimizer.evaluate_configuration()

# Step 5: Generate visualizations
figures = optimizer.generate_visualizations()

# Step 6: Generate report
report = optimizer.generate_report()
```

## Configuration Options

### OptimizationConfig

- `n_channels`: Number of detection channels (default: 3)
- `wavelength_bounds`: Wavelength range for filters (default: (500, 700))
- `bandwidth_bounds`: Bandwidth range for filters (default: (15.0, 50.0))
- `n_filter_trials`: Number of optimization trials (default: 200)
- `optimize_excitation`: Whether to optimize excitation wavelengths (default: True)
- `excitation_search_range`: Search range around peak excitation (default: 30.0)
- `n_training_samples`: Number of SBI training samples (default: 5000)
- `n_validation_samples`: Number of SBI validation samples (default: 1000)
- `min_r_squared`: Minimum R² threshold for good performance (default: 0.7)
- `save_results`: Whether to save results to files (default: True)
- `generate_plots`: Whether to generate visualization plots (default: True)
- `verbose`: Whether to print detailed output (default: True)

### ExperimentalConstraints

- `available_lasers`: List of available laser wavelengths (optional)
- `available_filters`: List of available filter center wavelengths (optional)
- `min_photon_budget`: Minimum photon budget (default: 100.0)
- `max_photon_budget`: Maximum photon budget (default: 1000.0)
- `background_level`: Background fluorescence level (default: 30.0)

## Command Line Options

```
usage: optimize_microscope_parameters.py [-h] [--spectra-dir SPECTRA_DIR]
                                        [--output-dir OUTPUT_DIR]
                                        [--fluorophores FLUOROPHORES [FLUOROPHORES ...]]
                                        [--auto-select] [--n-channels N_CHANNELS]
                                        [--n-filter-trials N_FILTER_TRIALS]
                                        [--no-excitation-opt]
                                        [--n-training-samples N_TRAINING_SAMPLES]
                                        [--n-validation-samples N_VALIDATION_SAMPLES]
                                        [--no-plots] [--no-save] [--quiet]

Optimize microscope parameters for multiplexed fluorescence imaging

optional arguments:
  -h, --help            show this help message and exit
  --spectra-dir SPECTRA_DIR
                        Directory containing fluorophore spectra files (default: data/spectra_npz)
  --output-dir OUTPUT_DIR
                        Directory to save optimization results (default: optimization_results)
  --fluorophores FLUOROPHORES [FLUOROPHORES ...]
                        List of fluorophore names to optimize (if not provided, interactive selection)
  --auto-select         Automatically select fluorophores instead of interactive selection
  --n-channels N_CHANNELS
                        Number of detection channels (default: 3)
  --n-filter-trials N_FILTER_TRIALS
                        Number of trials for filter optimization (default: 200)
  --no-excitation-opt   Skip excitation wavelength optimization
  --n-training-samples N_TRAINING_SAMPLES
                        Number of training samples for SBI (default: 5000)
  --n-validation-samples N_VALIDATION_SAMPLES
                        Number of validation samples for SBI (default: 1000)
  --no-plots            Skip plot generation
  --no-save             Don't save results to files
  --quiet               Reduce output verbosity
```

## Output Files

The optimization script generates several output files in the `optimization_results` directory:

### Generated Files

1. **optimization_report.json**: Complete optimization report with all results
2. **fluorophore_spectra.png**: Plot of fluorophore excitation and emission spectra
3. **detection_channels.png**: Plot of optimized detection filters with spectra overlay
4. **crosstalk_matrix.png**: Heatmap of excitation crosstalk matrix
5. **spectral_overlap.png**: Analysis of spectral overlap between fluorophores

### Report Structure

The JSON report contains:
- `timestamp`: When the optimization was run
- `configuration`: All optimization and experimental parameters
- `fluorophores`: List of fluorophores used
- `optimization_results`: Optimal filter and excitation configurations
- `performance_metrics`: SBI evaluation results including R² values and multiplexing capacity

## Key Features

### 1. Filter Optimization
Uses the `optimize_filter_configuration` method from `SBISimulator` to find optimal detection filter configurations by:
- Testing multiple random filter configurations
- Evaluating signal separation quality
- Selecting the configuration with the best performance score

### 2. Excitation Optimization
Uses the `find_optimal_excitation` function from `Microscope.py` to:
- Minimize crosstalk between fluorophores
- Maximize self-excitation efficiency
- Search around peak excitation wavelengths

### 3. Performance Evaluation
Uses SBI training to assess the quality of the optimized configuration:
- Trains a neural posterior estimator
- Evaluates concentration prediction accuracy
- Calculates R² values and multiplexing capacity metrics

### 4. Comprehensive Visualization
Generates multiple plots to visualize:
- Fluorophore spectra (excitation and emission)
- Optimized detection channels with spectral overlay
- Excitation crosstalk matrix
- Spectral overlap analysis

## Examples

Run the example script to see different usage patterns:

```bash
python example_optimization.py
```

This will run three different optimization examples:
1. Basic optimization with default settings
2. Custom optimization with advanced configuration
3. Step-by-step optimization showing individual steps

## Requirements

- Python 3.7+
- All dependencies from the multiplex_sim package
- SBI package for simulation-based inference
- Standard scientific Python packages (numpy, matplotlib, scipy, torch)

## Troubleshooting

### Common Issues

1. **Missing spectra files**: Ensure fluorophore spectra are available in `data/spectra_npz/`
2. **SBI import errors**: Install the SBI package with `pip install sbi`
3. **Memory issues**: Reduce `n_training_samples` and `n_validation_samples` for large problems
4. **Slow optimization**: Reduce `n_filter_trials` for faster (but potentially less optimal) results

### Performance Tips

- Use fewer training samples for quick testing
- Increase `n_filter_trials` for better optimization quality
- Use `--quiet` flag to reduce output verbosity
- Skip plots with `--no-plots` for faster execution

## Integration with Existing Code

This optimization script is designed to work seamlessly with your existing multiplex_sim codebase:

- Uses `SBISimulator.optimize_filter_configuration()` as the core optimization engine
- Integrates with `SBITrainer` for performance evaluation
- Leverages existing plotting functions from `multiplex_sim.plotting`
- Uses `find_optimal_excitation()` from `Microscope.py` for excitation optimization

The script provides a high-level interface that combines all these components into a unified optimization workflow.
