"""
Excitation analysis and visualization tools for multiplexed microscopy.

This module provides functions for analyzing excitation crosstalk, optimization results,
and comparing different excitation strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings

from .sbi_simulator import SBISimulator, SBIConfig


def plot_excitation_crosstalk_matrix(
    simulator: SBISimulator,
    excitation_wavelengths: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'viridis',
    annotate: bool = True
) -> plt.Figure:
    """
    Plot excitation crosstalk matrix as a heatmap.
    
    Args:
        simulator: SBI simulator instance
        excitation_wavelengths: Custom excitation wavelengths (if None, uses simulator's)
        figsize: Figure size
        cmap: Colormap for heatmap
        annotate: Whether to annotate cells with values
        
    Returns:
        Matplotlib figure
    """
    if excitation_wavelengths is None:
        excitation_wavelengths = simulator.excitation_wavelengths
    
    if excitation_wavelengths is None:
        raise ValueError("No excitation wavelengths available")
    
    crosstalk_matrix = simulator.calculate_excitation_crosstalk_matrix(excitation_wavelengths)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(crosstalk_matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Excitation Efficiency')
    
    # Set labels
    fluorophore_names = simulator.fluorophore_names
    ax.set_xticks(range(len(fluorophore_names)))
    ax.set_yticks(range(len(fluorophore_names)))
    ax.set_xticklabels(fluorophore_names, rotation=45)
    ax.set_yticklabels([f"{name}\n({wl:.0f} nm)" for name, wl in 
                       zip(fluorophore_names, excitation_wavelengths)])
    
    ax.set_xlabel("Fluorophore (Excited)")
    ax.set_ylabel("Laser (Excitation Wavelength)")
    ax.set_title("Excitation Crosstalk Matrix")
    
    # Add text annotations
    if annotate:
        for i in range(len(fluorophore_names)):
            for j in range(len(fluorophore_names)):
                text = ax.text(j, i, f'{crosstalk_matrix[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if crosstalk_matrix[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    return fig


def analyze_excitation_crosstalk(
    simulator: SBISimulator,
    excitation_wavelengths: Optional[List[float]] = None,
    threshold: float = 0.1
) -> Dict[str, Union[float, List[Tuple[str, str, float]]]]:
    """
    Analyze excitation crosstalk and return summary statistics.
    
    Args:
        simulator: SBI simulator instance
        excitation_wavelengths: Custom excitation wavelengths
        threshold: Minimum crosstalk value to report
        
    Returns:
        Dictionary with crosstalk analysis results
    """
    if excitation_wavelengths is None:
        excitation_wavelengths = simulator.excitation_wavelengths
    
    if excitation_wavelengths is None:
        raise ValueError("No excitation wavelengths available")
    
    crosstalk_matrix = simulator.calculate_excitation_crosstalk_matrix(excitation_wavelengths)
    fluorophore_names = simulator.fluorophore_names
    
    # Calculate statistics
    off_diagonal = crosstalk_matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    
    max_crosstalk = off_diagonal.max()
    mean_crosstalk = off_diagonal.mean()
    std_crosstalk = off_diagonal.std()
    
    # Find significant crosstalk pairs
    significant_crosstalk = []
    for i in range(len(fluorophore_names)):
        for j in range(len(fluorophore_names)):
            if i != j and crosstalk_matrix[i, j] > threshold:
                laser_name = fluorophore_names[i]
                fluor_name = fluorophore_names[j]
                efficiency = crosstalk_matrix[i, j]
                relative_efficiency = efficiency / crosstalk_matrix[i, i] * 100
                significant_crosstalk.append((laser_name, fluor_name, efficiency, relative_efficiency))
    
    # Sort by efficiency
    significant_crosstalk.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'max_crosstalk': max_crosstalk,
        'mean_crosstalk': mean_crosstalk,
        'std_crosstalk': std_crosstalk,
        'significant_crosstalk': significant_crosstalk,
        'crosstalk_matrix': crosstalk_matrix,
        'excitation_wavelengths': excitation_wavelengths
    }


def plot_excitation_spectra_with_lasers(
    simulator: SBISimulator,
    excitation_wavelengths: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_emission: bool = True
) -> plt.Figure:
    """
    Plot excitation and emission spectra with laser wavelengths marked.
    
    Args:
        simulator: SBI simulator instance
        excitation_wavelengths: Custom excitation wavelengths
        figsize: Figure size
        show_emission: Whether to show emission spectra
        
    Returns:
        Matplotlib figure
    """
    if excitation_wavelengths is None:
        excitation_wavelengths = simulator.excitation_wavelengths
    
    if excitation_wavelengths is None:
        raise ValueError("No excitation wavelengths available")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    wavelengths = simulator.spectra_manager.wavelengths
    fluorophore_names = simulator.fluorophore_names
    
    # Plot excitation spectra
    for i, name in enumerate(fluorophore_names):
        if name in simulator.excitation_spectra:
            excitation = simulator.excitation_spectra[name]
            ax.plot(wavelengths, excitation, '--', alpha=0.7, 
                   label=f"{name} Excitation", color=f'C{i}')
    
    # Plot emission spectra if requested
    if show_emission:
        for i, name in enumerate(fluorophore_names):
            emission = simulator.emission_spectra[name]
            ax.plot(wavelengths, emission, '-', alpha=0.7, 
                   label=f"{name} Emission", color=f'C{i}')
    
    # Mark laser wavelengths
    for i, (name, exc_wl) in enumerate(zip(fluorophore_names, excitation_wavelengths)):
        ax.axvline(exc_wl, color=f'C{i}', linestyle=':', linewidth=2, alpha=0.8,
                  label=f"{name} Laser ({exc_wl:.0f} nm)")
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("Fluorophore Spectra with Optimized Excitation Wavelengths")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_excitation_strategies(
    simulator: SBISimulator,
    strategies: Dict[str, List[float]],
    center_wavelengths: List[float],
    bandwidths: List[float],
    n_samples: int = 1000,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, Dict[str, Dict]]:
    """
    Compare different excitation strategies by simulation.
    
    Args:
        simulator: SBI simulator instance
        strategies: Dictionary mapping strategy names to excitation wavelengths
        center_wavelengths: Detection filter centers
        bandwidths: Detection filter bandwidths
        n_samples: Number of test samples
        figsize: Figure size
        
    Returns:
        Tuple of (figure, results dictionary)
    """
    results = {}
    
    # Generate test concentrations
    np.random.seed(42)
    test_concentrations = np.random.dirichlet(np.ones(len(simulator.fluorophore_names)), size=n_samples)
    
    # Test each strategy
    for strategy_name, exc_wavelengths in strategies.items():
        # Simulate signals
        signals = simulator.simulate_batch_with_excitation(
            test_concentrations, center_wavelengths, bandwidths, 
            excitation_wavelengths=exc_wavelengths, add_noise=False
        )
        
        # Calculate crosstalk analysis
        crosstalk_analysis = analyze_excitation_crosstalk(simulator, exc_wavelengths)
        
        # Calculate signal statistics
        signal_means = signals.mean(dim=0).numpy()
        signal_stds = signals.std(dim=0).numpy()
        signal_snr = signal_means / (signal_stds + 1e-6)
        
        results[strategy_name] = {
            'signals': signals,
            'signal_means': signal_means,
            'signal_stds': signal_stds,
            'signal_snr': signal_snr,
            'crosstalk_analysis': crosstalk_analysis,
            'excitation_wavelengths': exc_wavelengths
        }
    
    # Create comparison plots
    n_strategies = len(strategies)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Signal means comparison
    ax = axes[0, 0]
    x = np.arange(len(center_wavelengths))
    width = 0.8 / n_strategies
    
    for i, (strategy_name, result) in enumerate(results.items()):
        ax.bar(x + i * width, result['signal_means'], width, 
               label=strategy_name, alpha=0.7)
    
    ax.set_xlabel("Detection Channel")
    ax.set_ylabel("Mean Signal (photons)")
    ax.set_title("Signal Means by Strategy")
    ax.set_xticks(x + width * (n_strategies - 1) / 2)
    ax.set_xticklabels([f"Ch{i+1}" for i in range(len(center_wavelengths))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Signal-to-noise ratio
    ax = axes[0, 1]
    for i, (strategy_name, result) in enumerate(results.items()):
        ax.bar(x + i * width, result['signal_snr'], width, 
               label=strategy_name, alpha=0.7)
    
    ax.set_xlabel("Detection Channel")
    ax.set_ylabel("Signal-to-Noise Ratio")
    ax.set_title("Signal-to-Noise Ratio by Strategy")
    ax.set_xticks(x + width * (n_strategies - 1) / 2)
    ax.set_xticklabels([f"Ch{i+1}" for i in range(len(center_wavelengths))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Crosstalk comparison
    ax = axes[1, 0]
    strategy_names = list(strategies.keys())
    max_crosstalks = [results[name]['crosstalk_analysis']['max_crosstalk'] for name in strategy_names]
    mean_crosstalks = [results[name]['crosstalk_analysis']['mean_crosstalk'] for name in strategy_names]
    
    x_pos = np.arange(len(strategy_names))
    ax.bar(x_pos - 0.2, max_crosstalks, 0.4, label='Max Crosstalk', alpha=0.7)
    ax.bar(x_pos + 0.2, mean_crosstalks, 0.4, label='Mean Crosstalk', alpha=0.7)
    
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Crosstalk Efficiency")
    ax.set_title("Excitation Crosstalk Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategy_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Signal correlation matrix for best strategy
    best_strategy = min(strategy_names, key=lambda x: results[x]['crosstalk_analysis']['mean_crosstalk'])
    best_signals = results[best_strategy]['signals'].numpy()
    
    ax = axes[1, 1]
    corr_matrix = np.corrcoef(best_signals.T)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(center_wavelengths)))
    ax.set_yticks(range(len(center_wavelengths)))
    ax.set_xticklabels([f"Ch{i+1}" for i in range(len(center_wavelengths))])
    ax.set_yticklabels([f"Ch{i+1}" for i in range(len(center_wavelengths))])
    ax.set_title(f"Signal Correlation Matrix\n({best_strategy})")
    
    # Add correlation values
    for i in range(len(center_wavelengths)):
        for j in range(len(center_wavelengths)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center",
                   color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax)
    
    plt.suptitle("Excitation Strategy Comparison")
    plt.tight_layout()
    
    return fig, results


def plot_background_excitation_analysis(
    simulator: SBISimulator,
    excitation_wavelengths: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot background excitation analysis.
    
    Args:
        simulator: SBI simulator instance
        excitation_wavelengths: Custom excitation wavelengths
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if simulator.background_excitation is None:
        raise ValueError("Background excitation spectrum not available")
    
    if excitation_wavelengths is None:
        excitation_wavelengths = simulator.excitation_wavelengths
    
    if excitation_wavelengths is None:
        raise ValueError("No excitation wavelengths available")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot background excitation spectrum
    wavelengths = simulator.spectra_manager.wavelengths
    ax1.plot(wavelengths, simulator.background_excitation, 'k-', linewidth=2, 
             label=f'{simulator.config.background_fluorophore} Excitation')
    
    # Mark laser wavelengths
    fluorophore_names = simulator.fluorophore_names
    for i, (name, exc_wl) in enumerate(zip(fluorophore_names, excitation_wavelengths)):
        ax1.axvline(exc_wl, color=f'C{i}', linestyle='--', alpha=0.8, 
                   label=f"{name} Laser ({exc_wl:.0f} nm)")
    
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normalized Excitation")
    ax1.set_title(f"Background ({simulator.config.background_fluorophore}) Excitation Spectrum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate background response for different laser combinations
    n_lasers = len(excitation_wavelengths)
    laser_combinations = []
    labels = []
    
    # Individual lasers
    for i in range(n_lasers):
        combo = np.zeros(n_lasers)
        combo[i] = 1
        laser_combinations.append(combo)
        labels.append(f"Only {fluorophore_names[i]}")
    
    # All lasers
    laser_combinations.append(np.ones(n_lasers))
    labels.append("All Lasers")
    
    bg_responses = []
    for combo in laser_combinations:
        response = simulator.calculate_background_excitation_response(
            excitation_wavelengths, combo
        )
        bg_responses.append(response)
    
    # Plot background responses
    colors = [f'C{i}' for i in range(n_lasers)] + ['red']
    ax2.bar(range(len(bg_responses)), bg_responses, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel("Background Response")
    ax2.set_title("Background Excitation by Different Lasers")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_excitation_optimization_report(
    simulator: SBISimulator,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a comprehensive report on excitation optimization.
    
    Args:
        simulator: SBI simulator instance
        output_path: Path to save report (if None, returns as string)
        
    Returns:
        Report as string
    """
    if simulator.excitation_wavelengths is None:
        raise ValueError("No excitation wavelengths available")
    
    # Analyze crosstalk
    crosstalk_analysis = analyze_excitation_crosstalk(simulator)
    
    # Generate report
    report_lines = [
        "=" * 80,
        "EXCITATION OPTIMIZATION REPORT",
        "=" * 80,
        "",
        f"Fluorophores: {', '.join(simulator.fluorophore_names)}",
        f"Number of fluorophores: {len(simulator.fluorophore_names)}",
        f"Wavelength range: {simulator.config.wavelength_range[0]}-{simulator.config.wavelength_range[1]} nm",
        "",
        "OPTIMIZED EXCITATION WAVELENGTHS:",
        "-" * 40,
    ]
    
    for i, (name, wl) in enumerate(zip(simulator.fluorophore_names, simulator.excitation_wavelengths)):
        report_lines.append(f"{i+1:2d}. {name:<10}: {wl:6.1f} nm")
    
    report_lines.extend([
        "",
        "EXCITATION CROSSTALK ANALYSIS:",
        "-" * 40,
        f"Maximum crosstalk:     {crosstalk_analysis['max_crosstalk']:.3f}",
        f"Mean crosstalk:        {crosstalk_analysis['mean_crosstalk']:.3f}",
        f"Std deviation:         {crosstalk_analysis['std_crosstalk']:.3f}",
        "",
        "SIGNIFICANT CROSSTALK PAIRS (>10%):",
        "-" * 40,
    ])
    
    if crosstalk_analysis['significant_crosstalk']:
        for laser, fluor, efficiency, relative in crosstalk_analysis['significant_crosstalk']:
            report_lines.append(f"{laser} laser → {fluor}: {efficiency:.3f} ({relative:.1f}% of self)")
    else:
        report_lines.append("No significant crosstalk detected.")
    
    # Background analysis
    if simulator.background_excitation is not None:
        report_lines.extend([
            "",
            "BACKGROUND EXCITATION ANALYSIS:",
            "-" * 40,
        ])
        
        single_laser_bg = []
        for i in range(len(simulator.fluorophore_names)):
            combo = np.zeros(len(simulator.fluorophore_names))
            combo[i] = 1
            bg_response = simulator.calculate_background_excitation_response(
                simulator.excitation_wavelengths, combo
            )
            single_laser_bg.append(bg_response)
            report_lines.append(f"{simulator.fluorophore_names[i]} laser: {bg_response:.3f}")
        
        all_laser_bg = simulator.calculate_background_excitation_response(
            simulator.excitation_wavelengths, np.ones(len(simulator.fluorophore_names))
        )
        
        report_lines.extend([
            f"All lasers combined: {all_laser_bg:.3f}",
            f"Background range: {min(single_laser_bg):.3f} - {max(single_laser_bg):.3f}",
        ])
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "1. Use these optimized wavelengths for minimal crosstalk",
        "2. Consider laser power balancing to minimize background",
        "3. Validate with experimental measurements",
        "4. Monitor for spectral drift in long experiments",
        "",
        "=" * 80,
    ])
    
    report = "\n".join(report_lines)
    
    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report
