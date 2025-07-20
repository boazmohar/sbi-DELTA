"""
SBI Training module for multiplexed fluorescence microscopy.

This module provides high-level interfaces for training and evaluating 
simulation-based inference models for fluorophore concentration estimation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
import pickle
from dataclasses import dataclass, asdict
import json
from tqdm import tqdm

# SBI imports
try:
    from sbi import utils as sbi_utils
    from sbi.inference import SNPE, simulate_for_sbi
    from sbi.inference.posteriors.direct_posterior import DirectPosterior
    from sbi.utils.user_input_checks import process_prior, process_simulator
    from sbi.utils import BoxUniform
    SBI_AVAILABLE = True
except ImportError:
    SBI_AVAILABLE = False
    warnings.warn("SBI package not available. Install with: pip install sbi")

from .sbi_simulator import SBISimulator, SBIConfig, create_sbi_simulator
from .plotting import plot_simulation_results


class FlatPrior:
    """
    Custom flat prior for SBI that works with DirectPosterior and enable_transform=False.
    This is based on your FluorFilterPrior implementation.
    """
    
    def __init__(self, low: torch.Tensor, high: torch.Tensor, return_numpy: bool = False):
        """
        Initialize flat prior with bounds.
        
        Args:
            low: Lower bounds for each parameter
            high: Upper bounds for each parameter
            return_numpy: Whether to return numpy arrays (for compatibility)
        """
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.return_numpy = return_numpy
        self.event_shape = self.low.shape
        
    def sample(self, sample_shape=torch.Size([])):
        """Sample from uniform distribution within bounds."""
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        
        # Sample uniformly between bounds
        samples = torch.rand(n, *self.event_shape) * (self.high - self.low) + self.low
        
        if self.return_numpy:
            return samples.numpy()
        return samples
    
    def log_prob(self, values):
        """Calculate log probability (uniform within bounds, -inf outside)."""
        if self.return_numpy:
            values = torch.as_tensor(values, dtype=torch.float32)
        
        # Check if values are within bounds
        within_bounds = torch.all((values >= self.low) & (values <= self.high), dim=-1)
        
        # Log probability is constant within bounds, -inf outside
        log_volume = torch.sum(torch.log(self.high - self.low))
        log_probs = torch.where(within_bounds, -log_volume, torch.tensor(-float('inf')))
        
        if self.return_numpy:
            return log_probs.numpy()
        return log_probs
    
    @property
    def support(self):
        """Return the support of the distribution."""
        from torch.distributions.constraints import interval
        return interval(self.low, self.high)


@dataclass
class TrainingConfig:
    """Configuration for SBI training."""
    
    # Training parameters
    n_training_samples: int = 10000
    n_validation_samples: int = 1000
    
    # Training options
    learning_rate: float = 5e-4
    training_batch_size: int = 200
    validation_fraction: float = 0.1
    max_num_epochs: int = 2000000
    
    # Early stopping
    stop_after_epochs: int = 20
    
    # Device
    device: str = "cpu"  # or "cuda" if available


@dataclass
class ExperimentConfig:
    """Configuration for a complete SBI experiment."""
    
    # Fluorophores and spectra
    fluorophore_names: List[str] = None
    spectra_dir: str = "data/spectra_npz"
    
    # Filter configuration
    center_wavelengths: List[float] = None
    bandwidths: List[float] = None
    
    # Prior configuration
    prior_type: str = "dirichlet"  # "dirichlet", "beta", or "flat"
    prior_params: Dict[str, float] = None
    
    # SBI and training configs
    sbi_config: SBIConfig = None
    training_config: TrainingConfig = None
    
    # Experiment metadata
    experiment_name: str = "sbi_experiment"
    description: str = ""
    
    def __post_init__(self):
        if self.sbi_config is None:
            self.sbi_config = SBIConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
        if self.prior_params is None:
            self.prior_params = {"concentration": 1.0}


class SBITrainer:
    """
    High-level trainer for SBI models.
    """
    
    def __init__(self, config: ExperimentConfig):
        if not SBI_AVAILABLE:
            raise ImportError("SBI package is required. Install with: pip install sbi")
        
        self.config = config
        self.simulator = create_sbi_simulator(
            config.fluorophore_names,
            config.spectra_dir,
            config.sbi_config
        )
        
        # Training state
        self.prior = None
        self.inference = None
        self.posterior = None
        self.training_data = None
        self.validation_data = None
        self.training_history = {}
        
        # Results storage
        self.results = {}
        
    def setup_prior(self) -> torch.distributions.Distribution:
        """Set up the prior distribution."""
        if self.config.prior_type == "dirichlet":
            concentration = self.config.prior_params.get("concentration", 1.0)
            self.prior = self.simulator.create_dirichlet_prior(concentration)
        elif self.config.prior_type == "beta":
            alpha = self.config.prior_params.get("alpha", 1.0)
            beta = self.config.prior_params.get("beta", 1.0)
            # For independent Beta priors, we use BoxUniform as approximation
            n_fluors = len(self.config.fluorophore_names)
            self.prior = BoxUniform(
                low=torch.zeros(n_fluors),
                high=torch.ones(n_fluors)
            )
        elif self.config.prior_type == "flat":
            # Use custom flat prior
            n_fluors = len(self.config.fluorophore_names)
            low = torch.zeros(n_fluors)
            high = torch.ones(n_fluors)
            self.prior = FlatPrior(low, high, return_numpy=False)
        else:
            raise ValueError(f"Unknown prior type: {self.config.prior_type}")
        
        return self.prior
    
    def generate_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data."""
        print("Generating training data...")
        
        theta, x = self.simulator.generate_training_data(
            n_samples=self.config.training_config.n_training_samples,
            center_wavelengths=self.config.center_wavelengths,
            bandwidths=self.config.bandwidths,
            prior_type=self.config.prior_type,
            prior_params=self.config.prior_params
        )
        
        self.training_data = (theta, x)
        print(f"Generated {len(theta)} training samples")
        print(f"Parameter shape: {theta.shape}, Observation shape: {x.shape}")
        
        return theta, x
    
    def generate_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate validation data."""
        print("Generating validation data...")
        
        theta_val, x_val = self.simulator.generate_training_data(
            n_samples=self.config.training_config.n_validation_samples,
            center_wavelengths=self.config.center_wavelengths,
            bandwidths=self.config.bandwidths,
            prior_type=self.config.prior_type,
            prior_params=self.config.prior_params
        )
        
        self.validation_data = (theta_val, x_val)
        print(f"Generated {len(theta_val)} validation samples")
        
        return theta_val, x_val
    
    def train(self) -> DirectPosterior:
        """Train the SBI model."""
        if self.training_data is None:
            self.generate_training_data()
        
        if self.prior is None:
            self.setup_prior()
        
        theta, x = self.training_data
        
        print("Setting up SBI inference...")
        
        # Prepare simulator and prior for SBI
        def wrapped_simulator(params):
            return self.simulator.simulate_batch(
                params.numpy(),
                self.config.center_wavelengths,
                self.config.bandwidths,
                add_noise=True
            )
        prior, _, _ = process_prior(self.prior)
        simulator = process_simulator(wrapped_simulator, prior, is_numpy_simulator=False)
        
        
        
        # Initialize inference
        self.inference = SNPE(prior=prior)
        
        # Train the model
        print("Training neural posterior estimator...")
        
        # Add training data
        self.inference.append_simulations(theta, x)
        
        # Train with custom parameters
        density_estimator = self.inference.train(
            training_batch_size=self.config.training_config.training_batch_size,
            learning_rate=self.config.training_config.learning_rate,
            validation_fraction=self.config.training_config.validation_fraction,
            stop_after_epochs=self.config.training_config.stop_after_epochs,
            show_train_summary=True
        )
        
        # Build posterior using your approach with DirectPosterior and enable_transform=False
        print("Building posterior...")
        
        self.posterior = DirectPosterior(
            posterior_estimator=density_estimator,
            prior=prior,
            device=self.config.training_config.device,
            enable_transform=False  # Disable transform as in your implementation
        )
        
        print("Training completed!")
        return self.posterior
    
    def evaluate_on_validation(self, n_posterior_samples: int = 100) -> Dict[str, Any]:
        """Evaluate the trained model on validation data."""
        if self.posterior is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        if self.validation_data is None:
            self.generate_validation_data()
        
        theta_val, x_val = self.validation_data
        
        print(f"Evaluating on {len(theta_val)} validation samples...")
        
        # Collect predictions
        predictions = []
        r_squared_values = []
        
        for i in tqdm(range(len(x_val)), desc="Evaluating"):
            x_i = x_val[i]
            theta_true = theta_val[i].numpy()
            
            try:
                # Sample from posterior
                samples = self.posterior.sample(
                    (n_posterior_samples,), 
                    x=x_i, 
                    show_progress_bars=False
                ).numpy()
                
                # Calculate mean prediction
                theta_pred = samples.mean(axis=0)
                predictions.append(theta_pred)
                
                # Calculate R²
                r2 = self.simulator.calculate_r_squared(
                    theta_true.reshape(1, -1),
                    samples
                ).mean()
                r_squared_values.append(r2)
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate sample {i}: {e}")
                predictions.append(np.full_like(theta_true, np.nan))
                r_squared_values.append(np.nan)
        
        predictions = np.array(predictions)
        r_squared_values = np.array(r_squared_values)
        
        # Calculate evaluation metrics
        results = {
            "n_samples": len(theta_val),
            "mean_r_squared": np.nanmean(r_squared_values),
            "std_r_squared": np.nanstd(r_squared_values),
            "median_r_squared": np.nanmedian(r_squared_values),
            "r_squared_values": r_squared_values,
            "true_parameters": theta_val.numpy(),
            "predicted_parameters": predictions,
            "validation_observations": x_val.numpy()
        }
        
        # Calculate per-fluorophore metrics
        for i, name in enumerate(self.config.fluorophore_names):
            true_conc = theta_val[:, i].numpy()
            pred_conc = predictions[:, i]
            
            # Correlation
            valid_mask = ~(np.isnan(true_conc) | np.isnan(pred_conc))
            if valid_mask.sum() > 1:
                correlation = np.corrcoef(true_conc[valid_mask], pred_conc[valid_mask])[0, 1]
            else:
                correlation = np.nan
            
            results[f"{name}_correlation"] = correlation
            results[f"{name}_mae"] = np.nanmean(np.abs(true_conc - pred_conc))
            results[f"{name}_rmse"] = np.sqrt(np.nanmean((true_conc - pred_conc)**2))
        
        self.results["validation"] = results
        
        print(f"Validation Results:")
        print(f"  Mean R²: {results['mean_r_squared']:.3f} ± {results['std_r_squared']:.3f}")
        print(f"  Median R²: {results['median_r_squared']:.3f}")
        
        return results
    
    def analyze_multiplexing_capacity(
        self, 
        n_test_samples: int = 1000,
        r_squared_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Analyze the multiplexing capacity of the current configuration."""
        print("Analyzing multiplexing capacity...")
        
        # Generate test data
        theta_test, x_test = self.simulator.generate_training_data(
            n_samples=n_test_samples,
            center_wavelengths=self.config.center_wavelengths,
            bandwidths=self.config.bandwidths,
            prior_type=self.config.prior_type,
            prior_params=self.config.prior_params
        )
        
        # Evaluate on test data
        r_squared_values = []
        
        for i in tqdm(range(len(x_test)), desc="Testing"):
            x_i = x_test[i]
            theta_true = theta_test[i].numpy()
            
            try:
                samples = self.posterior.sample((50,), x=x_i, show_progress_bars=False).numpy()
                r2 = self.simulator.calculate_r_squared(
                    theta_true.reshape(1, -1), samples
                ).mean()
                r_squared_values.append(r2)
            except:
                r_squared_values.append(np.nan)
        
        r_squared_values = np.array(r_squared_values)
        
        # Analysis
        good_performance_fraction = np.mean(r_squared_values >= r_squared_threshold)
        
        results = {
            "n_fluorophores": len(self.config.fluorophore_names),
            "n_channels": len(self.config.center_wavelengths),
            "r_squared_threshold": r_squared_threshold,
            "good_performance_fraction": good_performance_fraction,
            "mean_r_squared": np.nanmean(r_squared_values),
            "r_squared_distribution": r_squared_values,
            "filter_configuration": {
                "center_wavelengths": self.config.center_wavelengths,
                "bandwidths": self.config.bandwidths
            }
        }
        
        self.results["multiplexing_capacity"] = results
        
        print(f"Multiplexing Capacity Analysis:")
        print(f"  {len(self.config.fluorophore_names)} fluorophores, {len(self.config.center_wavelengths)} channels")
        print(f"  {good_performance_fraction:.1%} of samples achieve R² ≥ {r_squared_threshold}")
        print(f"  Mean R²: {np.nanmean(r_squared_values):.3f}")
        
        return results
    
    def save_experiment(self, save_dir: Union[str, Path]) -> None:
        """Save the complete experiment."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = asdict(self.config)
        # Convert SBIConfig and TrainingConfig to dicts
        config_dict["sbi_config"] = asdict(self.config.sbi_config)
        config_dict["training_config"] = asdict(self.config.training_config)
        
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save posterior
        if self.posterior is not None:
            torch.save(self.posterior, save_dir / "posterior.pkl")
        
        # Save training data
        if self.training_data is not None:
            theta, x = self.training_data
            torch.save({"theta": theta, "x": x}, save_dir / "training_data.pkl")
        
        # Save validation data
        if self.validation_data is not None:
            theta_val, x_val = self.validation_data
            torch.save({"theta": theta_val, "x": x_val}, save_dir / "validation_data.pkl")
        
        # Save results
        if self.results:
            with open(save_dir / "results.pkl", "wb") as f:
                pickle.dump(self.results, f)
        
        print(f"Experiment saved to {save_dir}")
    
    @classmethod
    def load_experiment(cls, save_dir: Union[str, Path]) -> 'SBITrainer':
        """Load a saved experiment."""
        save_dir = Path(save_dir)
        
        # Load configuration
        with open(save_dir / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Reconstruct config objects
        sbi_config = SBIConfig(**config_dict["sbi_config"])
        training_config = TrainingConfig(**config_dict["training_config"])
        
        config_dict["sbi_config"] = sbi_config
        config_dict["training_config"] = training_config
        
        config = ExperimentConfig(**config_dict)
        
        # Create trainer
        trainer = cls(config)
        
        # Load posterior
        posterior_path = save_dir / "posterior.pkl"
        if posterior_path.exists():
            trainer.posterior = torch.load(posterior_path)
        
        # Load training data
        training_data_path = save_dir / "training_data.pkl"
        if training_data_path.exists():
            data = torch.load(training_data_path)
            trainer.training_data = (data["theta"], data["x"])
        
        # Load validation data
        validation_data_path = save_dir / "validation_data.pkl"
        if validation_data_path.exists():
            data = torch.load(validation_data_path)
            trainer.validation_data = (data["theta"], data["x"])
        
        # Load results
        results_path = save_dir / "results.pkl"
        if results_path.exists():
            with open(results_path, "rb") as f:
                trainer.results = pickle.load(f)
        
        return trainer


def run_sbi_experiment(
    fluorophore_names: List[str],
    center_wavelengths: List[float],
    bandwidths: List[float],
    experiment_name: str = "sbi_experiment",
    spectra_dir: str = "data/spectra_npz",
    save_dir: Optional[str] = None,
    **kwargs
) -> SBITrainer:
    """
    Convenience function to run a complete SBI experiment.
    
    Args:
        fluorophore_names: List of fluorophore names
        center_wavelengths: Center wavelengths for detection filters
        bandwidths: Bandwidths for detection filters
        experiment_name: Name for the experiment
        spectra_dir: Directory containing spectra files
        save_dir: Directory to save results (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Trained SBITrainer instance
    """
    # Create configuration
    config = ExperimentConfig(
        fluorophore_names=fluorophore_names,
        center_wavelengths=center_wavelengths,
        bandwidths=bandwidths,
        experiment_name=experiment_name,
        spectra_dir=spectra_dir,
        **kwargs
    )
    
    # Create and run trainer
    trainer = SBITrainer(config)
    
    # Setup and train
    trainer.setup_prior()
    trainer.train()
    
    # Evaluate
    trainer.evaluate_on_validation()
    trainer.analyze_multiplexing_capacity()
    
    # Save if requested
    if save_dir is not None:
        trainer.save_experiment(save_dir)
    
    return trainer
