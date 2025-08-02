from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Dirichlet, Uniform, Distribution
from typing import Optional
from .config import PriorConfig, BaseConfig
from .stick_breaking_prior import StickBreakingPrior
import matplotlib.pyplot as plt
import numpy as np

class PriorManager:

    """
    Manages creation of priors for SBI-DELTA using PriorConfig and BaseConfig.
    """
    def __init__(self, prior_config: PriorConfig, base_config: BaseConfig, excitation_manager=None):
        self.config = prior_config
        self.base_config = base_config
        self.excitation_manager = excitation_manager
        self.n_fluorophores = len(base_config.dye_names)
        
        # Validation
        if self.config.include_background_ratio and not getattr(self.base_config, 'bg_dye', None):
            raise ValueError("If include_background_ratio is True, you must set bg_dye in BaseConfig.")
            
        if self.config.include_filter_params and excitation_manager is None:
            raise ValueError("If include_filter_params is True, you must provide an excitation_manager.")

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

    def get_filter_prior(self) -> Optional[Distribution]:
        """Returns prior distribution for filter parameters using stick breaking process."""
        if not self.config.include_filter_params:
            return None
            
        # Calculate total available wavelength range
        total_width = self.base_config.max_wavelength - self.base_config.min_wavelength
        
        # Create stick breaking prior for filter and gap widths
        return StickBreakingPrior(
            n_filters=self.config.n_filters,
            total_width=total_width,
            min_filter_width=self.config.min_filter_width,
            max_filter_width=self.config.max_filter_width,
            concentration=1.0  # Can be made configurable if needed
        )
        
        return MultipleIndependent(filter_priors)

    def get_joint_prior(self) -> Distribution:
        """
        Returns a joint prior over concentrations, background ratio (optional),
        and filter parameters (optional).
        """
        concentration_prior = self.get_concentration_prior()
        background_prior = self.get_background_ratio_prior()
        filter_prior = self.get_filter_prior()
        
        # Combine all enabled priors
        class JointPrior(Distribution):
            arg_constraints = {}
            
            def sample(self, sample_shape=torch.Size()):
                samples = [concentration_prior.sample(sample_shape)]
                
                if background_prior is not None:
                    b = background_prior.sample(sample_shape)
                    if len(b.shape) == len(sample_shape):
                        b = b.unsqueeze(-1)
                    samples.append(b)
                    
                if filter_prior is not None:
                    f = filter_prior.sample(sample_shape)
                    samples.append(f)
                    
                return torch.cat(samples, dim=-1)
                
            def log_prob(self, theta):
                # Split parameters
                n_conc = concentration_prior.event_shape[0]
                c = theta[..., :n_conc]
                cur_idx = n_conc
                
                log_probs = [concentration_prior.log_prob(c)]
                
                if background_prior is not None:
                    b = theta[..., cur_idx]
                    log_probs.append(background_prior.log_prob(b))
                    cur_idx += 1
                    
                if filter_prior is not None:
                    f = theta[..., cur_idx:]
                    log_probs.append(filter_prior.log_prob(f))
                    
                return sum(log_probs)
                
        return JointPrior()
    
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
