# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sbi-DELTA is a simulation-based inference framework for designing and optimizing multiplexed fluorescence microscopy experiments. The project uses neural posterior estimation to infer fluorophore concentrations from simulated spectral measurements and optimize microscope parameters for maximum multiplexing capacity.

## Environment Setup

This project uses conda/mamba for environment management:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate sbi_env

# Install additional dependencies if needed
pip install sbi  # For simulation-based inference
```

## Key Package Structure

### Core Modules

- `sbi_delta/`: Main package containing the refactored architecture
  - `config.py`: Configuration dataclasses (BaseConfig, FilterConfig, ExcitationConfig, PriorConfig)
  - `simulator/`: Emission simulation components
    - `emission_simulator.py`: Core emission simulation logic
    - `base_simulator.py`: Base simulator interface
  - `trainer.py`: SBI training functionality
  - `spectra_manager.py`: Fluorophore spectra loading and management
  - `filter_bank.py`: Detection filter implementations
  - `excitation_manager.py`: Excitation wavelength optimization
  - `prior_manager.py`: Prior distribution management

### Legacy/Reference Modules

- `multiplex_sim/`: Original implementation (being refactored)
  - `sbi_simulator.py`: Original simulator implementation
  - `sbi_trainer.py`: Original training implementation
  - `Microscope.py`: Microscope parameter optimization
  - `plotting.py`: Visualization utilities

### Data

- `data/spectra/`: Fluorophore absorption/emission spectra (CSV/Excel)
- `data/spectra_npz/`: Preprocessed spectra in NumPy format

## Development Workflow

### Running Tests

Tests are scattered across both main directory and subdirectories. Run all tests with:

```bash
# Run tests in main directory
python -m pytest test_*.py -v

# Run tests in sbi_delta subdirectory
python -m pytest sbi_delta/test_*.py -v

# Run tests in multiplex_sim subdirectory
python -m pytest multiplex_sim/test_*.py -v
```

### Running Notebooks

Jupyter notebooks are organized in:
- `notebooks/`: Main analysis and demo notebooks
- `sbi_delta/notebooks/`: Package-specific notebooks

Start Jupyter from the project root:

```bash
jupyter lab
```

### Key Scripts

- `optimize_microscope_parameters.py`: Complete microscope optimization workflow
- `example_optimization.py`: Example usage patterns

## Architecture Notes

### Configuration System

The project uses a dataclass-based configuration system in `sbi_delta/config.py`:
- `BaseConfig`: Core simulation parameters (wavelength range, photon budget)
- `FilterConfig`: Detection filter configuration
- `ExcitationConfig`: Excitation wavelength settings
- `PriorConfig`: Prior distribution parameters

### Simulation Pipeline

1. **Spectra Loading**: `SpectraManager` loads fluorophore spectra from NPZ files
2. **Simulation Setup**: Configure parameters via config dataclasses
3. **Emission Simulation**: `EmissionSimulator` generates synthetic measurements
4. **Training**: `Trainer` class handles SBI neural posterior estimation
5. **Optimization**: Optimize filters/excitation for maximum multiplexing capacity

### SBI Integration

The project uses the `sbi` package for simulation-based inference:
- Neural Posterior Estimation (NPE) with normalizing flows
- Custom priors for fluorophore concentrations (Dirichlet distribution)
- DirectPosterior for inference without MCMC

## Code Style and Patterns

- Extensive use of type hints throughout the codebase
- Dataclasses for configuration management
- Path objects from `pathlib` for file handling
- NumPy arrays for spectral data, PyTorch tensors for SBI
- Comprehensive docstrings following NumPy style

## Logging and Debugging

The project uses Python logging with hierarchical loggers. TensorBoard logs are saved in:
- `notebooks/sbi-logs/`: Training logs from notebook experiments
- `sbi_delta/notebooks/sbi-logs/`: Logs from sbi_delta experiments

## Performance Considerations

- Spectral data is cached in NPZ format for faster loading
- SBI training can be computationally intensive - adjust sample counts accordingly
- Filter optimization uses random search with configurable trial counts
- Background fluorescence modeling is optional but impacts performance