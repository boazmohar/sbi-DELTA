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
from tqdm import tqdm

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
        if hasattr(simulator, 'prior_manager') and simulator.prior_manager is not None:
            self.prior = simulator.prior_manager.get_joint_prior()
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
            training_batch_size=256,
            learning_rate=0.0005,
            validation_fraction=0.1,
            stop_after_epochs=20,
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
        posterior_width = []
        for i in tqdm(range(self.n_val), desc='Validating', leave=True):
            samples = self.posterior.sample((1000,), x=val_x[i], show_progress_bars=False)
            mean = samples.mean(dim=0).numpy()
            std = samples.std(dim=0).numpy()
            cv = np.divide(std, mean, out=np.zeros_like(std), where=mean!=0)
            pred_theta.append(mean)
            posterior_width.append(cv)
        pred_theta = np.stack(pred_theta)
        posterior_width = np.stack(posterior_width)
        # Per-example R^2 and RMSE
        r2_scores = []
        rmse_scores = []
        for i in range(self.n_val):
            r2 = r2_score(val_theta[i].numpy(), pred_theta[i])
            rmse = np.sqrt(np.mean((val_theta[i].numpy() - pred_theta[i]) ** 2))
            r2_scores.append(r2)
            rmse_scores.append(rmse)
        # Aggregate RMSE
        rmse = np.sqrt(np.mean((val_theta.numpy() - pred_theta) ** 2))
        self.results['val_theta'] = val_theta
        self.results['val_x'] = val_x
        self.results['pred_theta'] = pred_theta
        self.results['posterior_width'] = posterior_width
        self.results['r2_scores'] = r2_scores
        self.results['rmse_scores'] = rmse_scores
        self.results['rmse'] = rmse
        print(f"Validation mean R^2: {np.mean(r2_scores):.3f}, RMSE: {rmse:.4f}")
        return r2_scores, rmse_scores, rmse

    def plot_pred_vs_true(self, idx=None, ax=None):
        """
        Scatter plot of predicted vs true parameters for a given validation example (or all), with dye/background names.
        """
        import matplotlib.pyplot as plt
        val_theta = self.results.get('val_theta')
        pred_theta = self.results.get('pred_theta')
        if val_theta is None or pred_theta is None:
            print("Run validate() first.")
            return
        val_theta = val_theta.numpy()
        pred_theta = pred_theta
        n_params = val_theta.shape[1]
        # Get dye names from config
        dye_names = list(self.simulator.config.dye_names)
        if hasattr(self.simulator.config, 'bg_dye') and self.simulator.config.bg_dye:
            dye_names = dye_names + [self.simulator.config.bg_dye]
        # Pad dye_names if needed
        if len(dye_names) < n_params:
            dye_names += [f'Param {i+1}' for i in range(len(dye_names), n_params)]
        if idx is not None:
            val_theta = val_theta[idx:idx+1]
            pred_theta = pred_theta[idx:idx+1]
        if ax is None:
            fig, ax = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        if n_params == 1:
            ax = [ax]
        for i in range(n_params):
            ax[i].scatter(val_theta[:, i], pred_theta[:, i], alpha=0.5, label=dye_names[i])
            ax[i].plot([val_theta[:, i].min(), val_theta[:, i].max()],
                      [val_theta[:, i].min(), val_theta[:, i].max()], 'r--')
            ax[i].set_xlabel(f'True {dye_names[i]}')
            ax[i].set_ylabel(f'Predicted {dye_names[i]}')
            ax[i].set_title(f'{dye_names[i]}')
            ax[i].legend()
        plt.tight_layout()
        return ax

    def plot_error_histograms(self, ax=None):
        """
        Plot histograms of per-example errors (true - predicted) for each parameter, with dye/background names.
        """
        import matplotlib.pyplot as plt
        val_theta = self.results.get('val_theta')
        pred_theta = self.results.get('pred_theta')
        if val_theta is None or pred_theta is None:
            print("Run validate() first.")
            return
        val_theta = val_theta.numpy()
        pred_theta = pred_theta
        errors = val_theta - pred_theta
        n_params = errors.shape[1]
        dye_names = list(self.simulator.config.dye_names)
        if hasattr(self.simulator.config, 'bg_dye') and self.simulator.config.bg_dye:
            dye_names = dye_names + [self.simulator.config.bg_dye]
        if len(dye_names) < n_params:
            dye_names += [f'Param {i+1}' for i in range(len(dye_names), n_params)]
        if ax is None:
            fig, ax = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        if n_params == 1:
            ax = [ax]
        for i in range(n_params):
            ax[i].hist(errors[:, i], bins=30, alpha=0.7, label=dye_names[i])
            ax[i].set_xlabel(f'Error (True - Pred) {dye_names[i]}')
            ax[i].set_ylabel('Count')
            ax[i].set_title(f'{dye_names[i]} Error')
            ax[i].legend()
        plt.tight_layout()
        return ax

    def plot_r2_rmse_distributions(self, ax=None):
        """
        Plot histograms of per-example R^2 and RMSE scores.
        """
        import matplotlib.pyplot as plt
        r2_scores = self.results.get('r2_scores')
        rmse_scores = self.results.get('rmse_scores')
        if r2_scores is None or rmse_scores is None:
            print("Run validate() first.")
            return
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(r2_scores, bins=np.linspace(-1, 1, 50), alpha=0.7)
        ax[0].set_xlabel('R^2 Score')
        ax[0].set_ylabel('Count')
        ax[0].set_title('Per-example R^2')
        ax[0].set_xlim(-1, 1)
        ax[1].hist(rmse_scores, bins=30, alpha=0.7)
        ax[1].set_xlabel('RMSE')
        ax[1].set_ylabel('Count')
        ax[1].set_title('Per-example RMSE')
        plt.tight_layout()
        return ax

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.results, os.path.join(self.save_dir, "results.pt"))
        print(f"Experiment results saved to {self.save_dir}/results.pt")
