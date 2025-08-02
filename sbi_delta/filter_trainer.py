"""Training wrapper for filter optimization."""

import os
import torch
import numpy as np
from pathlib import Path
from sbi import inference as inference

class FilterTrainer:
    """
    Training wrapper for filter optimization using SBI.
    Similar to the original Trainer but handles combined concentration and filter parameters.
    """
    
    def __init__(
        self,
        simulator,
        prior_manager,
        training_batch_size=100,
        num_workers=4,
        save_dir="filter_optimization_results",
    ):
        """
        Initialize trainer.
        
        Args:
            simulator: FilterPriorSimulator instance
            prior_manager: PriorManager instance
            training_batch_size: Batch size for training
            num_workers: Number of parallel workers for simulation
            save_dir: Directory for saving results
        """
        self.simulator = simulator
        self.prior = prior_manager.get_joint_prior()
        self.save_dir = Path(save_dir)
        self.training_batch_size = training_batch_size
        self.num_workers = num_workers
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize inference objects
        self.inference = None
        self.posterior = None
        self._theta = None
        self._x = None
        
    def simulate_for_sbi(self, n_simulations):
        """Generate simulation data for training."""
        # Sample parameters from prior
        self._theta = self.prior.sample((n_simulations,))
        
        # Simulate for each parameter set
        x = []
        for theta in self._theta:
            counts = self.simulator.simulate(params=theta.numpy(), add_noise=True)
            x.append(counts.flatten())
        self._x = torch.tensor(np.stack(x), dtype=torch.float32)
        
    def train_density_estimator(self):
        """Train the neural density estimator."""
        if self._theta is None or self._x is None:
            raise ValueError("No training data available. Run simulate_for_sbi first.")
            
        # Initialize inference with default SNPE
        self.inference = inference.SNPE(prior=self.prior)
            
        # Train
        density_estimator = self.inference.train(
            training_batch_size=self.training_batch_size,
            x=self._x,
            theta=self._theta,
        )
        
        return density_estimator
        
    def build_posterior(self, x):
        """Build posterior for given observation."""
        if self.inference is None:
            raise ValueError("No trained inference object available.")
            
        return self.inference.build_posterior()