from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Dirichlet, Uniform, Distribution
from typing import Optional
from .config import PriorConfig, BaseConfig
import matplotlib.pyplot as plt


class PriorManager:

    """
    Manages creation of priors for SBI-DELTA using PriorConfig and BaseConfig.
    """
    def __init__(self, prior_config: PriorConfig, base_config: BaseConfig):
        self.config = prior_config
        self.base_config = base_config
        self.n_fluorophores = len(base_config.dye_names)
        # Validation: if background ratio is requested, bg_dye must be set
        if self.config.include_background_ratio and not getattr(self.base_config, 'bg_dye', None):
            raise ValueError("If include_background_ratio is True, you must set bg_dye in BaseConfig.")

    def get_concentration_prior(self) -> Distribution:
        # Dirichlet prior over fluorophore concentrations, using config value
        conc = self.config.dirichlet_concentration
        return Dirichlet(conc * torch.ones(self.n_fluorophores))
    def visualize_dirichlet_prior(self, n_samples=1000, ax=None):
        """
        Visualize samples from the Dirichlet prior as a histogram for each fluorophore.
        """
        prior = self.get_concentration_prior()
        samples = prior.sample((n_samples,)).numpy()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(samples.shape[1]):
            ax.hist(samples[:, i], bins=30, alpha=0.6, label=f'Fluor {i+1}', density=True)
        ax.set_title(f"Dirichlet Prior (concentration={self.config.dirichlet_concentration})")
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Density")
        ax.legend()
        return ax
    
    def __repr__(self):
        return (f"PriorManager(n_fluorophores={self.n_fluorophores}, "
                f"dye_names={list(self.base_config.dye_names)}, "
                f"bg_dye={self.base_config.bg_dye!r}, "
                f"dirichlet_concentration={self.config.dirichlet_concentration}, "
                f"include_background_ratio={self.config.include_background_ratio}, "
                f"background_ratio_bounds={self.config.background_ratio_bounds})")

    def get_background_ratio_prior(self) -> Optional[Distribution]:
        if self.config.include_background_ratio:
            low, high = self.config.background_ratio_bounds
            return Uniform(low, high)
        return None

    def get_joint_prior(self) -> Distribution:
        """
        Returns a joint prior over concentrations and (optionally) background ratio.
        The background ratio is a value in [0,1] representing the fraction of total photons
        allocated to background.
        """
        concentration_prior = self.get_concentration_prior()
        background_prior = self.get_background_ratio_prior()
        if background_prior is not None:
            # Compose arg_constraints from the underlying priors
            class JointPrior(Distribution):
                arg_constraints = { }
                def sample(self, sample_shape=torch.Size()):
                    c = concentration_prior.sample(sample_shape)
                    b = background_prior.sample(sample_shape)
                    if len(b.shape) == len(sample_shape):
                        b = b.unsqueeze(-1)
                    return torch.cat([c, b], dim=-1)
                def log_prob(self, theta):
                    # theta shape: (..., n_fluorophores + 1)
                    c = theta[..., :concentration_prior.event_shape[0]]
                    b = theta[..., -1]
                    lp_c = concentration_prior.log_prob(c)
                    lp_b = background_prior.log_prob(b)
                    return lp_c + lp_b
            return JointPrior()
        else:
            # Use a custom wrapper class for Dirichlet, matching the JointPrior interface
            concentration_prior = self.get_concentration_prior()
            class DirichletPrior(Distribution):
                arg_constraints = {}
               
                def sample(self, sample_shape=torch.Size()):
                    return concentration_prior.sample(sample_shape)
                def log_prob(self, theta):
                    return concentration_prior.log_prob(theta)
            return DirichletPrior()
    
    def visualize_joint_prior(self, n_samples=1000, ax=None):
        """
        Visualize samples from the joint prior as histograms for each parameter.
        """
        joint_prior = self.get_joint_prior()
        samples = joint_prior.sample((n_samples,)).numpy()
        n_params = samples.shape[1]
        if ax is None:
            fig, ax = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        if n_params == 1:
            ax = [ax]
        dye_names = list(self.base_config.dye_names)
        for i in range(self.n_fluorophores):
            ax[i].hist(samples[:, i], bins=30, alpha=0.7, color='C0', density=True)
            title = dye_names[i] if i < len(dye_names) else f'Concentration {i+1}'
            ax[i].set_title(title)
            ax[i].set_xlabel('Fraction')
            ax[i].set_ylabel('Density')
        if n_params > self.n_fluorophores:
            ax[-1].hist(samples[:, -1], bins=30, alpha=0.7, color='C1', density=True)
            ax[-1].set_title('Background Ratio')
            ax[-1].set_xlabel('Ratio')
            ax[-1].set_ylabel('Density')
        return ax
