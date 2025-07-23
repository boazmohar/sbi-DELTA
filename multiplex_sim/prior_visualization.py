"""
Prior visualization utilities for SBI simulator with filters.

This module provides plotting functions to visualize prior distributions
and parameter exploration for the enhanced SBI simulator.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from .sbi_simulator_with_filters import CustomFlatPrior, EnhancedSBISimulator


class PriorVisualizer:
    """
    Visualization utilities for prior distributions and parameter exploration.
    """
    
    def __init__(self, simulator: EnhancedSBISimulator):
        self.simulator = simulator
        self.prior = None
    
    def set_prior(self, prior: CustomFlatPrior):
        """Set the prior for visualization."""
        self.prior = prior
    
    def plot_prior_distributions(self, n_samples: int = 1000, figsize: Tuple[int, int] = (15, 12)):
        """
        Plot the prior distributions for all parameter types.
        
        Args:
            n_samples: Number of samples to generate for plotting
            figsize: Figure size for the plots
        """
        if self.prior is None:
            raise ValueError("Please set a prior using set_prior() before plotting")
        
        # Generate samples
        samples = self.prior.sample((n_samples,))
        params = self.prior.extract_parameters(samples)
        
        # Determine number of subplots needed
        n_plots = 3 + (1 if self.prior.include_background else 0)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Plot 1: Concentration distributions
        concentrations = params['concentrations'].numpy()
        for i in range(concentrations.shape[1]):
            axes[0].hist(concentrations[:, i], bins=30, alpha=0.7, label=f'Fluor {i+1}', density=True)
        axes[0].set_title('Concentration Parameters Distribution')
        axes[0].set_xlabel('Concentration')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        
        # Plot 2: Center wavelength distributions
        centers = params['center_wavelengths'].numpy()
        for i in range(centers.shape[1]):
            axes[1].hist(centers[:, i], bins=30, alpha=0.7, label=f'Channel {i+1}', density=True)
        axes[1].set_title('Filter Center Wavelengths Distribution')
        axes[1].set_xlabel('Center Wavelength (nm)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        
        # Plot 3: Bandwidth distributions
        bandwidths = params['bandwidths'].numpy()
        for i in range(bandwidths.shape[1]):
            axes[2].hist(bandwidths[:, i], bins=30, alpha=0.7, label=f'Channel {i+1}', density=True)
        axes[2].set_title('Filter Bandwidths Distribution')
        axes[2].set_xlabel('Bandwidth (nm)')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        
        # Plot 4: Background parameter (if included)
        if self.prior.include_background and len(axes) > 3:
            background = params['background'].numpy()
            axes[3].hist(background, bins=30, alpha=0.7, color='purple', density=True)
            axes[3].set_title('Background Parameter Distribution')
            axes[3].set_xlabel('Background Amplitude')
            axes[3].set_ylabel('Density')
        
        # Hide any extra axes
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_correlations(self, n_samples: int = 1000, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation matrix between different parameter types.
        
        Args:
            n_samples: Number of samples to generate
            figsize: Figure size for the plot
        """
        if self.prior is None:
            raise ValueError("Please set a prior using set_prior() before plotting")
        
        # Generate samples
        samples = self.prior.sample((n_samples,))
        params = self.prior.extract_parameters(samples)
        
        # Flatten all parameters into a single array
        all_params = []
        param_names = []
        
        # Concentrations
        concentrations = params['concentrations'].numpy()
        for i in range(concentrations.shape[1]):
            all_params.append(concentrations[:, i])
            param_names.append(f'Conc_{i+1}')
        
        # Center wavelengths
        centers = params['center_wavelengths'].numpy()
        for i in range(centers.shape[1]):
            all_params.append(centers[:, i])
            param_names.append(f'Center_{i+1}')
        
        # Bandwidths
        bandwidths = params['bandwidths'].numpy()
        for i in range(bandwidths.shape[1]):
            all_params.append(bandwidths[:, i])
            param_names.append(f'BW_{i+1}')
        
        # Background
        if self.prior.include_background:
            background = params['background'].numpy()
            all_params.append(background[:, 0])
            param_names.append('Background')
        
        # Create correlation matrix
        data = np.array(all_params).T
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   xticklabels=param_names, yticklabels=param_names, ax=ax)
        ax.set_title('Parameter Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def plot_parameter_pairs(self, n_samples: int = 500, figsize: Tuple[int, int] = (15, 12)):
        """
        Plot pairwise relationships between parameters.
        
        Args:
            n_samples: Number of samples to generate
            figsize: Figure size for the plot
        """
        if self.prior is None:
            raise ValueError("Please set a prior using set_prior() before plotting")
        
        # Generate samples
        samples = self.prior.sample((n_samples,))
        params = self.prior.extract_parameters(samples)
        
        # Create DataFrame-like structure for seaborn
        data = {}
        
        # Concentrations
        concentrations = params['concentrations'].numpy()
        for i in range(concentrations.shape[1]):
            data[f'Conc_{i+1}'] = concentrations[:, i]
        
        # Center wavelengths
        centers = params['center_wavelengths'].numpy()
        for i in range(centers.shape[1]):
            data[f'Center_{i+1}'] = centers[:, i]
        
        # Bandwidths
        bandwidths = params['bandwidths'].numpy()
        for i in range(bandwidths.shape[1]):
            data[f'BW_{i+1}'] = bandwidths[:, i]
        
        # Background
        if self.prior.include_background:
            background = params['background'].numpy()
            data['Background'] = background[:, 0]
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Plot pairplot
        fig = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
        fig.fig.suptitle('Parameter Pairwise Relationships', y=1.02)
        return fig
    
    def plot_filter_configuration_space(self, n_samples: int = 1000, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the filter configuration space (center vs bandwidth).
        
        Args:
            n_samples: Number of samples to generate
            figsize: Figure size for the plot
        """
        if self.prior is None:
            raise ValueError("Please set a prior using set_prior() before plotting")
        
        # Generate samples
        samples = self.prior.sample((n_samples,))
        params = self.prior.extract_parameters(samples)
        
        centers = params['center_wavelengths'].numpy()
        bandwidths = params['bandwidths'].numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: All channels scatter
        for i in range(centers.shape[1]):
            axes[0].scatter(centers[:, i], bandwidths[:, i], alpha=0.5, label=f'Channel {i+1}')
        axes[0].set_xlabel('Center Wavelength (nm)')
        axes[0].set_ylabel('Bandwidth (nm)')
        axes[0].set_title('Filter Configuration Space')
        axes[0].legend()
        
        # Plot 2: Channel 1 vs Channel 2
        if centers.shape[1] >= 2:
            axes[1].scatter(centers[:, 0], centers[:, 1], alpha=0.5)
            axes[1].set_xlabel('Channel 1 Center (nm)')
            axes[1].set_ylabel('Channel 2 Center (nm)')
            axes[1].set_title('Center Wavelength Correlation')
        
        # Plot 3: Bandwidth distribution
        axes[2].hist(bandwidths.flatten(), bins=30, alpha=0.7, color='green')
        axes[2].set_xlabel('Bandwidth (nm)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Overall Bandwidth Distribution')
        
        # Plot 4: Center wavelength distribution
        axes[3].hist(centers.flatten(), bins=30, alpha=0.7, color='orange')
        axes[3].set_xlabel('Center Wavelength (nm)')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Overall Center Wavelength Distribution')
        
        plt.tight_layout()
        return fig


def create_demo_plots(simulator: EnhancedSBISimulator, fluorophore_names: list, save_path: Optional[str] = None):
    """
    Create a comprehensive demo of prior visualization.
    
    Args:
        simulator: Enhanced SBI simulator instance
        fluorophore_names: List of fluorophore names
        save_path: Optional path to save the plots
    """
    # Create prior
    prior = simulator.create_custom_prior()
    
    # Create visualizer
    visualizer = PriorVisualizer(simulator)
    visualizer.set_prior(prior)
    
    # Create plots
    fig1 = visualizer.plot_prior_distributions(n_samples=2000)
    fig2 = visualizer.plot_parameter_correlations(n_samples=1000)
    fig3 = visualizer.plot_filter_configuration_space(n_samples=1000)
    
    if save_path:
        fig1.savefig(f"{save_path}_prior_distributions.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{save_path}_correlations.png", dpi=300, bbox_inches='tight')
        fig3.savefig(f"{save_path}_filter_space.png", dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}_*.png")
    
    plt.show()
    return visualizer


if __name__ == "__main__":
    # Demo usage
    from .sbi_simulator_with_filters import create_enhanced_sbi_simulator
    
    # Create simulator
    fluorophore_names = ['AF488', 'AF555', 'AF594', 'AF647']
    simulator = create_enhanced_sbi_simulator(fluorophore_names)
    
    # Create demo plots
    visualizer = create_demo_plots(simulator, fluorophore_names)
