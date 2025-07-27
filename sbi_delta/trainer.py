"""
SBI Trainer script using sbi_delta configs, managers, and emission simulator.
This script can be run as a standalone entry point for SBI training and evaluation.
"""
import os
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi_delta.config import BaseConfig, ExcitationConfig, FilterConfig, PriorConfig
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.filter_bank import FilterBank
from sbi_delta.prior_manager import PriorManager
from sbi_delta.simulator.emission_simulator import EmissionSimulator
from sklearn.metrics import r2_score


# Trainer class for SBI workflow
class Trainer:
    def __init__(self, simulator, n_train=2000, n_val=500, save_dir="sbi_training_demo_results"):
        """
        Args:
            simulator: An instance of EmissionSimulator (or compatible simulator)
            n_train: Number of training samples
            n_val: Number of validation samples
            save_dir: Directory to save results
        """
        self.simulator = simulator
        # Get n_dyes from simulator config
        self.n_dyes = len(simulator.config.dye_names)
        self.n_train = n_train
        self.n_val = n_val
        self.save_dir = save_dir
        # Get prior from simulator if available, else fallback to BoxUniform
        if hasattr(simulator, 'prior') and simulator.prior is not None:
            self.prior = simulator.prior
        else:
            self.prior = BoxUniform(low=torch.zeros(self.n_dyes), high=torch.ones(self.n_dyes))
        self.posterior = None
        self.results = {}

    def generate_training_data(self):
        train_theta = self.prior.sample((self.n_train,))
        train_x = []
        for theta in train_theta:
            counts = self.simulator.simulate(concentrations=theta.numpy(), add_noise=True, debug=False)
            train_x.append(counts.flatten())
        train_x = torch.tensor(np.stack(train_x), dtype=torch.float32)
        self.results['train_theta'] = train_theta
        self.results['train_x'] = train_x
        return train_theta, train_x

    def train(self):
        train_theta, train_x = self.generate_training_data()
        inference = SNPE(prior=self.prior)
        inference.append_simulations(train_theta, train_x)
        density_estimator = inference.train(
            training_batch_size=128,
            learning_rate=5e-4,
            validation_fraction=0.1,
            stop_after_epochs=10,
            show_train_summary=True
        )
        self.posterior = inference.build_posterior(density_estimator)
        return self.posterior

    def validate(self):
        val_theta = self.prior.sample((self.n_val,))
        val_x = []
        for theta in val_theta:
            counts = self.simulator.simulate(concentrations=theta.numpy(), add_noise=True, debug=False)
            val_x.append(counts.flatten())
        val_x = torch.tensor(np.stack(val_x), dtype=torch.float32)
        pred_theta = []
        for i in range(self.n_val):
            samples = self.posterior.sample((100,), x=val_x[i])
            pred_theta.append(samples.mean(dim=0).numpy())
        pred_theta = np.stack(pred_theta)
        r2_scores = [r2_score(val_theta[i].numpy(), pred_theta[i]) for i in range(self.n_val)]
        # RMSE calculation
        rmse = np.sqrt(np.mean((val_theta.numpy() - pred_theta) ** 2))
        self.results['val_theta'] = val_theta
        self.results['val_x'] = val_x
        self.results['pred_theta'] = pred_theta
        self.results['r2_scores'] = r2_scores
        self.results['rmse'] = rmse
        print(f"Validation mean R^2: {np.mean(r2_scores):.3f}, RMSE: {rmse:.4f}")
        return r2_scores, rmse

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.results, os.path.join(self.save_dir, "results.pt"))
        print(f"Experiment results saved to {self.save_dir}/results.pt")
