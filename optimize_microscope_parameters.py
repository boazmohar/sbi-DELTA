#!/usr/bin/env python3
"""
Complete Microscope Parameter Optimization Script

This script provides a comprehensive solution for designing optimal microscope parameters
using the SBI simulator's optimize_filter_configuration function and related tools.

Usage:
    python optimize_microscope_parameters.py

Features:
- Interactive fluorophore selection
- Filter configuration optimization
- Excitation wavelength optimization
- Performance evaluation using SBI
- Comprehensive reporting and visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Import from the multiplex_sim package
from multiplex_sim.sbi_simulator import SBISimulator, SBIConfig, create_sbi_simulator
from multiplex_sim.sbi_trainer import SBITrainer, ExperimentConfig, TrainingConfig
from multiplex_sim.Microscope import find_optimal_excitation, MicroscopeConfig
from multiplex_sim.io import list_fluorophores
from multiplex_sim.plotting import (
    plot_fluorophores, plot_detection_channels, plot_crosstalk_matrix,
    plot_simulation_results, plot_spectral_overlap
)


@dataclass
class OptimizationConfig:
    """Configuration for microscope parameter optimization."""
    
    # Filter optimization parameters
    n_channels: int = 3
    wavelength_bounds: Tuple[float, float] = (500, 700)
    bandwidth_bounds: Tuple[float, float] = (15.0, 50.0)
    n_filter_trials: int = 200
    
    # Excitation optimization parameters
    optimize_excitation: bool = True
    excitation_search_range: float = 30.0
    
    # Evaluation parameters
    n_training_samples: int = 5000
    n_validation_samples: int = 1000
    n_evaluation_trials: int = 100
    
    # Performance thresholds
    min_r_squared: float = 0.7
    max_crosstalk: float = 0.3
    
    # Output settings
    save_results: bool = True
    generate_plots: bool = True
    verbose: bool = True


@dataclass
class ExperimentalConstraints:
    """Real-world experimental constraints."""
    
    # Available laser lines (nm)
    available_lasers: Optional[List[float]] = None
    
    # Available filter center wavelengths (nm)
    available_filters: Optional[List[float]] = None
    
    # Photon budget constraints
    min_photon_budget: float = 100.0
    max_photon_budget: float = 1000.0
    
    # Background constraints
    background_level: float = 30.0


class MicroscopeOptimizer:
    """Main class for microscope parameter optimization."""
    
    def __init__(
        self,
        spectra_dir: Union[str, Path] = "data/spectra_npz",
        optimization_config: Optional[OptimizationConfig] = None,
        experimental_constraints: Optional[ExperimentalConstraints] = None
    ):
        self.spectra_dir = Path(spectra_dir)
        self.opt_config = optimization_config or OptimizationConfig()
        self.constraints = experimental_constraints or ExperimentalConstraints()
        
        # Results storage
        self.results = {}
        self.fluorophore_names = []
        self.simulator = None
        self.trainer = None
        
        # Optimization results
        self.optimal_filters = None
        self.optimal_excitation = None
        self.performance_metrics = None
        
    def list_available_fluorophores(self) -> List[str]:
        """List all available fluorophores in the spectra directory."""
        return list_fluorophores(str(self.spectra_dir))
    
    def select_fluorophores_interactive(self) -> List[str]:
        """Interactive fluorophore selection."""
        available = self.list_available_fluorophores()
        
        if not available:
            raise FileNotFoundError(f"No fluorophore spectra found in {self.spectra_dir}")
        
        print("\nAvailable fluorophores:")
        for i, name in enumerate(available, 1):
            print(f"  {i:2d}. {name}")
        
        print(f"\nSelect fluorophores for optimization (e.g., '1,3,5' or '1-5'):")
        print(f"Available: {len(available)} fluorophores")
        
        while True:
            try:
                selection = input("Enter selection: ").strip()
                
                if not selection:
                    # Default selection - first few fluorophores
                    selected_indices = list(range(min(3, len(available))))
                elif '-' in selection:
                    # Range selection
                    start, end = map(int, selection.split('-'))
                    selected_indices = list(range(start-1, min(end, len(available))))
                else:
                    # Individual selection
                    selected_indices = [int(x.strip())-1 for x in selection.split(',')]
                
                # Validate indices
                selected_indices = [i for i in selected_indices if 0 <= i < len(available)]
                
                if not selected_indices:
                    print("Invalid selection. Please try again.")
                    continue
                
                selected_names = [available[i] for i in selected_indices]
                
                print(f"\nSelected fluorophores: {selected_names}")
                confirm = input("Confirm selection? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes', '']:
                    return selected_names
                    
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")
    
    def select_fluorophores_automatic(self, n_fluorophores: int = 3) -> List[str]:
        """Automatic fluorophore selection based on spectral separation."""
        available = self.list_available_fluorophores()
        
        if len(available) <= n_fluorophores:
            return available
        
        # Simple selection: evenly spaced fluorophores
        indices = np.linspace(0, len(available)-1, n_fluorophores, dtype=int)
        return [available[i] for i in indices]
    
    def setup_simulator(self, fluorophore_names: List[str]) -> SBISimulator:
        """Set up the SBI simulator with selected fluorophores."""
        self.fluorophore_names = fluorophore_names
        
        # Configure SBI parameters
        sbi_config = SBIConfig(
            wavelength_range=(450, 750),
            wavelength_step=1.0,
            total_dye_photons=self.constraints.max_photon_budget * 0.8,
            total_background_photons=self.constraints.background_level,
            optimize_excitation=self.opt_config.optimize_excitation,
            excitation_search_range=self.opt_config.excitation_search_range,
            include_excitation_crosstalk=True
        )
        
        self.simulator = create_sbi_simulator(
            fluorophore_names, 
            self.spectra_dir, 
            sbi_config
        )
        
        if self.opt_config.verbose:
            print(f"✓ Simulator configured for {len(fluorophore_names)} fluorophores")
            if self.simulator.excitation_wavelengths:
                print(f"✓ Excitation wavelengths: {self.simulator.excitation_wavelengths}")
        
        return self.simulator
    
    def optimize_detection_filters(self) -> Dict[str, Any]:
        """Optimize detection filter configuration using the SBI simulator."""
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized. Call setup_simulator() first.")
        
        if self.opt_config.verbose:
            print(f"\nOptimizing detection filters...")
            print(f"  Channels: {self.opt_config.n_channels}")
            print(f"  Wavelength bounds: {self.opt_config.wavelength_bounds}")
            print(f"  Bandwidth bounds: {self.opt_config.bandwidth_bounds}")
            print(f"  Trials: {self.opt_config.n_filter_trials}")
        
        # Use the optimize_filter_configuration method from SBISimulator
        filter_config = self.simulator.optimize_filter_configuration(
            n_channels=self.opt_config.n_channels,
            wavelength_bounds=self.opt_config.wavelength_bounds,
            bandwidth_bounds=self.opt_config.bandwidth_bounds,
            n_trials=self.opt_config.n_filter_trials
        )
        
        self.optimal_filters = filter_config
        
        if self.opt_config.verbose:
            print(f"✓ Optimal filter configuration found:")
            print(f"  Center wavelengths: {[f'{w:.1f}' for w in filter_config['center_wavelengths']]}")
            print(f"  Bandwidths: {[f'{b:.1f}' for b in filter_config['bandwidths']]}")
            print(f"  Optimization score: {filter_config['score']:.4f}")
        
        return filter_config
    
    def optimize_excitation_wavelengths(self) -> Dict[str, float]:
        """Optimize excitation wavelengths using existing optimization."""
        if not self.fluorophore_names:
            raise RuntimeError("Fluorophores not selected. Call setup_simulator() first.")
        
        if self.opt_config.verbose:
            print(f"\nOptimizing excitation wavelengths...")
            print(f"  Search range: ±{self.opt_config.excitation_search_range} nm")
        
        # Use the existing find_optimal_excitation function
        optimal_excitation = find_optimal_excitation(
            self.fluorophore_names,
            self.spectra_dir,
            search_range=self.opt_config.excitation_search_range
        )
        
        self.optimal_excitation = optimal_excitation
        
        if self.opt_config.verbose:
            print(f"✓ Optimal excitation wavelengths:")
            for name, wavelength in optimal_excitation.items():
                print(f"  {name}: {wavelength} nm")
        
        return optimal_excitation
    
    def evaluate_configuration(self) -> Dict[str, Any]:
        """Evaluate the optimized configuration using SBI training."""
        if self.optimal_filters is None:
            raise RuntimeError("Filters not optimized. Call optimize_detection_filters() first.")
        
        if self.opt_config.verbose:
            print(f"\nEvaluating configuration with SBI training...")
            print(f"  Training samples: {self.opt_config.n_training_samples}")
            print(f"  Validation samples: {self.opt_config.n_validation_samples}")
        
        # Set up experiment configuration
        experiment_config = ExperimentConfig(
            fluorophore_names=self.fluorophore_names,
            center_wavelengths=self.optimal_filters['center_wavelengths'],
            bandwidths=self.optimal_filters['bandwidths'],
            spectra_dir=str(self.spectra_dir),
            prior_type="dirichlet",
            prior_params={"concentration": 1.0},
            sbi_config=self.simulator.config,
            training_config=TrainingConfig(
                n_training_samples=self.opt_config.n_training_samples,
                n_validation_samples=self.opt_config.n_validation_samples,
                device="cpu"
            )
        )
        
        # Create and train SBI model
        self.trainer = SBITrainer(experiment_config)
        
        try:
            # Train the model
            self.trainer.setup_prior()
            self.trainer.train()
            
            # Evaluate performance
            validation_results = self.trainer.evaluate_on_validation(n_posterior_samples=50)
            multiplexing_results = self.trainer.analyze_multiplexing_capacity(
                n_test_samples=self.opt_config.n_evaluation_trials,
                r_squared_threshold=self.opt_config.min_r_squared
            )
            
            # Combine results
            performance_metrics = {
                "validation": validation_results,
                "multiplexing": multiplexing_results,
                "configuration": {
                    "filters": self.optimal_filters,
                    "excitation": self.optimal_excitation,
                    "fluorophores": self.fluorophore_names
                }
            }
            
            self.performance_metrics = performance_metrics
            
            if self.opt_config.verbose:
                print(f"✓ Evaluation completed:")
                print(f"  Mean R²: {validation_results['mean_r_squared']:.3f}")
                print(f"  Good performance fraction: {multiplexing_results['good_performance_fraction']:.1%}")
            
            return performance_metrics
            
        except Exception as e:
            warnings.warn(f"SBI evaluation failed: {e}")
            return {"error": str(e)}
    
    def generate_visualizations(self, save_dir: Optional[Path] = None) -> Dict[str, plt.Figure]:
        """Generate comprehensive visualizations of the optimization results."""
        if save_dir is None:
            save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        figures = {}
        
        try:
            # 1. Fluorophore spectra
            fig_spectra = plot_fluorophores(
                self.fluorophore_names, 
                str(self.spectra_dir),
                figsize=(12, 6)
            )
            figures["spectra"] = fig_spectra
            if self.opt_config.save_results:
                fig_spectra.savefig(save_dir / "fluorophore_spectra.png", dpi=300, bbox_inches='tight')
            
            # 2. Detection channels with spectra overlay
            if self.optimal_filters:
                # Create emission spectra dict for overlay
                emission_spectra = {}
                for name in self.fluorophore_names:
                    try:
                        data = np.load(self.spectra_dir / f"{name}.npz")
                        wl_em = data["wavelengths_emission"]
                        em = data["emission"]
                        emission_spectra[name] = (wl_em, em)
                    except Exception as e:
                        warnings.warn(f"Could not load emission spectrum for {name}: {e}")
                
                # Create filter visualization
                wavelengths = self.simulator.spectra_manager.wavelengths
                filters = self.simulator.filter_bank.create_filters(
                    self.optimal_filters['center_wavelengths'],
                    self.optimal_filters['bandwidths']
                )
                
                fig_channels = plot_detection_channels(
                    wavelengths, filters, 
                    self.optimal_filters['center_wavelengths'],
                    emission_spectra, figsize=(14, 8)
                )
                figures["detection_channels"] = fig_channels
                if self.opt_config.save_results:
                    fig_channels.savefig(save_dir / "detection_channels.png", dpi=300, bbox_inches='tight')
            
            # 3. Crosstalk matrix
            if self.optimal_excitation and self.simulator.excitation_wavelengths:
                crosstalk_matrix = self.simulator.calculate_excitation_crosstalk_matrix()
                fig_crosstalk = plot_crosstalk_matrix(
                    crosstalk_matrix, self.fluorophore_names,
                    self.simulator.excitation_wavelengths, figsize=(10, 8)
                )
                figures["crosstalk"] = fig_crosstalk
                if self.opt_config.save_results:
                    fig_crosstalk.savefig(save_dir / "crosstalk_matrix.png", dpi=300, bbox_inches='tight')
            
            # 4. Spectral overlap analysis
            fig_overlap = plot_spectral_overlap(
                self.fluorophore_names, str(self.spectra_dir), 
                spectrum_type="emission", figsize=(10, 8)
            )
            figures["spectral_overlap"] = fig_overlap
            if self.opt_config.save_results:
                fig_overlap.savefig(save_dir / "spectral_overlap.png", dpi=300, bbox_inches='tight')
            
            if self.opt_config.verbose:
                print(f"✓ Generated {len(figures)} visualization plots")
                if self.opt_config.save_results:
                    print(f"✓ Plots saved to {save_dir}")
            
            return figures
            
        except Exception as e:
            warnings.warn(f"Visualization generation failed: {e}")
            return {}
    
    def generate_report(self, save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if save_dir is None:
            save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        # Compile complete results
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "optimization_config": asdict(self.opt_config),
                "experimental_constraints": asdict(self.constraints),
                "spectra_directory": str(self.spectra_dir)
            },
            "fluorophores": self.fluorophore_names,
            "optimization_results": {
                "optimal_filters": self.optimal_filters,
                "optimal_excitation": self.optimal_excitation
            },
            "performance_metrics": self.performance_metrics
        }
        
        # Save report
        if self.opt_config.save_results:
            with open(save_dir / "optimization_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            if self.opt_config.verbose:
                print(f"✓ Report saved to {save_dir / 'optimization_report.json'}")
        
        return report
    
    def run_complete_optimization(
        self, 
        fluorophore_names: Optional[List[str]] = None,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """Run complete optimization workflow."""
        
        print("=" * 60)
        print("MICROSCOPE PARAMETER OPTIMIZATION")
        print("=" * 60)
        
        # Step 1: Select fluorophores
        if fluorophore_names is None:
            if interactive:
                fluorophore_names = self.select_fluorophores_interactive()
            else:
                fluorophore_names = self.select_fluorophores_automatic()
        
        # Step 2: Setup simulator
        self.setup_simulator(fluorophore_names)
        
        # Step 3: Optimize detection filters
        self.optimize_detection_filters()
        
        # Step 4: Optimize excitation wavelengths
        if self.opt_config.optimize_excitation:
            self.optimize_excitation_wavelengths()
        
        # Step 5: Evaluate configuration
        self.evaluate_configuration()
        
        # Step 6: Generate visualizations
        if self.opt_config.generate_plots:
            self.generate_visualizations()
        
        # Step 7: Generate report
        report = self.generate_report()
        
        # Print summary
        self.print_optimization_summary()
        
        return report
    
    def print_optimization_summary(self):
        """Print a summary of optimization results."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        print(f"Fluorophores: {', '.join(self.fluorophore_names)}")
        
        if self.optimal_filters:
            print(f"Optimal Filter Centers: {[f'{w:.1f}' for w in self.optimal_filters['center_wavelengths']]} nm")
            print(f"Optimal Filter Bandwidths: {[f'{b:.1f}' for b in self.optimal_filters['bandwidths']]} nm")
            print(f"Filter Optimization Score: {self.optimal_filters['score']:.4f}")
        
        if self.optimal_excitation:
            print("Optimal Excitation Wavelengths:")
            for name, wl in self.optimal_excitation.items():
                print(f"  {name}: {wl} nm")
        
        if self.performance_metrics and "validation" in self.performance_metrics:
            val_results = self.performance_metrics["validation"]
            print(f"Mean R²: {val_results['mean_r_squared']:.3f} ± {val_results['std_r_squared']:.3f}")
            
            if "multiplexing" in self.performance_metrics:
                mult_results = self.performance_metrics["multiplexing"]
                print(f"Good Performance Fraction: {mult_results['good_performance_fraction']:.1%}")
        
        print("=" * 60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize microscope parameters for multiplexed fluorescence imaging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument(
        "--spectra-dir", 
        type=str, 
        default="data/spectra_npz",
        help="Directory containing fluorophore spectra files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Directory to save optimization results"
    )
    
    # Fluorophore selection
    parser.add_argument(
        "--fluorophores",
        type=str,
        nargs="+",
        help="List of fluorophore names to optimize (if not provided, interactive selection)"
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically select fluorophores instead of interactive selection"
    )
    
    # Optimization parameters
    parser.add_argument(
        "--n-channels",
        type=int,
        default=3,
        help="Number of detection channels"
    )
    parser.add_argument(
        "--n-filter-trials",
        type=int,
        default=200,
        help="Number of trials for filter optimization"
    )
    parser.add_argument(
        "--no-excitation-opt",
        action="store_true",
        help="Skip excitation wavelength optimization"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n-training-samples",
        type=int,
        default=5000,
        help="Number of training samples for SBI"
    )
    parser.add_argument(
        "--n-validation-samples",
        type=int,
        default=1000,
        help="Number of validation samples for SBI"
    )
    
    # Output control
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser


def main():
    """Main function for command line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create optimization configuration
    opt_config = OptimizationConfig(
        n_channels=args.n_channels,
        n_filter_trials=args.n_filter_trials,
        optimize_excitation=not args.no_excitation_opt,
        n_training_samples=args.n_training_samples,
        n_validation_samples=args.n_validation_samples,
        save_results=not args.no_save,
        generate_plots=not args.no_plots,
        verbose=not args.quiet
    )
    
    # Create experimental constraints (can be extended based on args)
    constraints = ExperimentalConstraints()
    
    # Create optimizer
    optimizer = MicroscopeOptimizer(
        spectra_dir=args.spectra_dir,
        optimization_config=opt_config,
        experimental_constraints=constraints
    )
    
    try:
        # Run optimization
        report = optimizer.run_complete_optimization(
            fluorophore_names=args.fluorophores,
            interactive=not args.auto_select and args.fluorophores is None
        )
        
        print(f"\n✓ Optimization completed successfully!")
        if opt_config.save_results:
            print(f"✓ Results saved to {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
