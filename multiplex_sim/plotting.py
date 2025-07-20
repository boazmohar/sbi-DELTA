"""
Plotting utilities for multiplex simulation visualization.

This module provides functions for visualizing fluorophore spectra, detection channels,
crosstalk matrices, and simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from typing import List, Optional, Dict, Tuple, Union
import warnings


def plot_fluorophores(
    names: List[str], 
    npz_folder: Union[str, Path] = "spectra_npz", 
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot excitation (dashed) and emission (solid) spectra for multiple dyes.

    Args:
        names: List of dye names (without .npz extension)
        npz_folder: Folder where .npz spectra files are stored
        normalize: If True, peak-normalize each spectrum to max = 1
        figsize: Figure size (width, height)
        show_legend: Whether to show the legend
        
    Returns:
        matplotlib Figure object
    """
    folder = Path(npz_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder '{npz_folder}' does not exist.")

    n = len(names)
    colormap = cm.get_cmap("turbo", n)

    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, dye_name in enumerate(names):
        path = folder / f"{dye_name}.npz"
        if not path.exists():
            warnings.warn(f"Missing file: {path.name}")
            continue

        data = np.load(path)
        wl_em = data["wavelengths_emission"]
        em = data["emission"]
        wl_ex = data["wavelengths_excitation"]
        ex = data["excitation"]

        if normalize:
            if em.max() > 0:
                em = em / em.max()
            if ex.max() > 0:
                ex = ex / ex.max()

        color = colormap(idx)
        ax.plot(wl_em, em, label=f"{dye_name} Emission", color=color, linewidth=2)
        ax.plot(wl_ex, ex, '--', label=f"{dye_name} Excitation", color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Normalized Intensity" if normalize else "Raw Intensity", fontsize=12)
    ax.set_title("Excitation and Emission Spectra", fontsize=14, fontweight='bold')
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_crosstalk_matrix(
    crosstalk_matrix: np.ndarray,
    fluor_names: List[str],
    excitation_wavelengths: List[float],
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot crosstalk matrix as a heatmap.
    
    Args:
        crosstalk_matrix: Matrix of crosstalk values
        fluor_names: List of fluorophore names
        excitation_wavelengths: List of excitation wavelengths
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create labels with wavelengths
    x_labels = [f"{name}\n({wl:.0f} nm)" for name, wl in zip(fluor_names, excitation_wavelengths)]
    y_labels = fluor_names
    
    # Plot heatmap
    im = ax.imshow(crosstalk_matrix, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(fluor_names)))
    ax.set_yticks(range(len(fluor_names)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Excitation Efficiency', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(fluor_names)):
        for j in range(len(fluor_names)):
            text = ax.text(j, i, f'{crosstalk_matrix[i, j]:.3f}',
                          ha="center", va="center", 
                          color="white" if crosstalk_matrix[i, j] < 0.5 else "black")
    
    ax.set_title("Crosstalk Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Excitation Laser", fontsize=12)
    ax.set_ylabel("Fluorophore Response", fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_detection_channels(
    wavelengths: np.ndarray,
    channel_filters: np.ndarray,
    center_wavelengths: List[float],
    emission_spectra: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot detection channel filters and optionally overlay emission spectra.
    
    Args:
        wavelengths: Wavelength array
        channel_filters: Array of channel filter responses
        center_wavelengths: List of channel center wavelengths
        emission_spectra: Optional dict of emission spectra to overlay
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot channel filters
    n_channels = len(center_wavelengths)
    colors = cm.get_cmap("Set1", n_channels)
    
    for i, center_wl in enumerate(center_wavelengths):
        color = colors(i)
        ax.fill_between(wavelengths, channel_filters[i], alpha=0.3, color=color,
                       label=f"Channel {i+1} ({center_wl:.0f} nm)")
        ax.plot(wavelengths, channel_filters[i], color=color, linewidth=2)
    
    # Overlay emission spectra if provided
    if emission_spectra:
        for name, (wl_em, em) in emission_spectra.items():
            em_normalized = em / em.max() if em.max() > 0 else em
            ax.plot(wl_em, em_normalized, '--', linewidth=2, alpha=0.8, label=f"{name} Emission")
    
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Filter Response / Normalized Emission", fontsize=12)
    ax.set_title("Detection Channel Filters", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_simulation_results(
    true_concentrations: np.ndarray,
    detected_signals: np.ndarray,
    dye_names: List[str],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot simulation results comparing true concentrations with detected signals.
    
    Args:
        true_concentrations: Array of true dye concentrations
        detected_signals: Array of detected signals
        dye_names: List of dye names
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    n_samples, n_dyes = true_concentrations.shape
    n_channels = detected_signals.shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: True concentrations heatmap
    im1 = axes[0].imshow(true_concentrations[:50].T, aspect='auto', cmap='viridis')
    axes[0].set_title("True Dye Concentrations", fontweight='bold')
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Dye")
    axes[0].set_yticks(range(n_dyes))
    axes[0].set_yticklabels(dye_names)
    plt.colorbar(im1, ax=axes[0], label='Concentration')
    
    # Plot 2: Detected signals heatmap
    im2 = axes[1].imshow(detected_signals[:50].T, aspect='auto', cmap='plasma')
    axes[1].set_title("Detected Signals", fontweight='bold')
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Detection Channel")
    axes[1].set_yticks(range(n_channels))
    axes[1].set_yticklabels([f"Ch {i+1}" for i in range(n_channels)])
    plt.colorbar(im2, ax=axes[1], label='Signal Intensity')
    
    # Plot 3: Correlation between total concentration and total signal
    total_conc = true_concentrations.sum(axis=1)
    total_signal = detected_signals.sum(axis=1)
    
    axes[2].scatter(total_conc, total_signal, alpha=0.6, s=20)
    axes[2].set_xlabel("Total True Concentration")
    axes[2].set_ylabel("Total Detected Signal")
    axes[2].set_title("Signal vs Concentration", fontweight='bold')
    
    # Add correlation coefficient
    corr_coef = np.corrcoef(total_conc, total_signal)[0, 1]
    axes[2].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[2].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_optimization_progress(
    optimization_history: List[float],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot optimization progress over iterations.
    
    Args:
        optimization_history: List of cost function values over iterations
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(optimization_history, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost Function Value", fontsize=12)
    ax.set_title("Excitation Wavelength Optimization Progress", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = np.argmin(optimization_history)
    best_value = optimization_history[best_idx]
    ax.plot(best_idx, best_value, 'ro', markersize=8, label=f'Best: {best_value:.4f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_spectral_overlap(
    dye_names: List[str],
    npz_folder: Union[str, Path] = "spectra_npz",
    spectrum_type: str = "emission",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot spectral overlap matrix between fluorophores.
    
    Args:
        dye_names: List of dye names
        npz_folder: Folder containing spectra files
        spectrum_type: "emission" or "excitation"
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    folder = Path(npz_folder)
    
    # Load spectra
    spectra = {}
    for name in dye_names:
        path = folder / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            if spectrum_type == "emission":
                wl = data["wavelengths_emission"]
                spectrum = data["emission"]
            else:
                wl = data["wavelengths_excitation"]
                spectrum = data["excitation"]
            
            # Normalize
            if spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()
            
            spectra[name] = (wl, spectrum)
    
    # Calculate overlap matrix
    n_dyes = len(dye_names)
    overlap_matrix = np.zeros((n_dyes, n_dyes))
    
    for i, name_i in enumerate(dye_names):
        for j, name_j in enumerate(dye_names):
            if name_i in spectra and name_j in spectra:
                wl_i, spec_i = spectra[name_i]
                wl_j, spec_j = spectra[name_j]
                
                # Interpolate to common wavelength grid
                common_wl = np.linspace(max(wl_i.min(), wl_j.min()), 
                                      min(wl_i.max(), wl_j.max()), 1000)
                spec_i_interp = np.interp(common_wl, wl_i, spec_i)
                spec_j_interp = np.interp(common_wl, wl_j, spec_j)
                
                # Calculate overlap (normalized dot product)
                overlap = np.trapz(spec_i_interp * spec_j_interp, common_wl)
                overlap_matrix[i, j] = overlap
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(overlap_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(n_dyes))
    ax.set_yticks(range(n_dyes))
    ax.set_xticklabels(dye_names, rotation=45, ha='right')
    ax.set_yticklabels(dye_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{spectrum_type.capitalize()} Spectral Overlap', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(n_dyes):
        for j in range(n_dyes):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.3f}',
                          ha="center", va="center",
                          color="white" if overlap_matrix[i, j] > 0.5 else "black")
    
    ax.set_title(f"{spectrum_type.capitalize()} Spectral Overlap Matrix", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_multiplexing_capacity(
    n_dyes_range: range,
    crosstalk_values: List[float],
    signal_quality: List[float],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot multiplexing capacity analysis showing crosstalk vs signal quality.
    
    Args:
        n_dyes_range: Range of number of dyes tested
        crosstalk_values: List of average crosstalk values for each n_dyes
        signal_quality: List of signal quality metrics for each n_dyes
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot crosstalk vs number of dyes
    ax1.plot(n_dyes_range, crosstalk_values, 'ro-', linewidth=2, markersize=6)
    ax1.set_xlabel("Number of Dyes", fontsize=12)
    ax1.set_ylabel("Average Crosstalk", fontsize=12)
    ax1.set_title("Crosstalk vs Multiplexing Level", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot signal quality vs number of dyes
    ax2.plot(n_dyes_range, signal_quality, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel("Number of Dyes", fontsize=12)
    ax2.set_ylabel("Signal Quality", fontsize=12)
    ax2.set_title("Signal Quality vs Multiplexing Level", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_photon_budget_analysis(
    photon_budgets: List[float],
    snr_values: List[float],
    detection_accuracy: List[float],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot analysis of photon budget effects on detection performance.
    
    Args:
        photon_budgets: List of photon budget values tested
        snr_values: Signal-to-noise ratio for each budget
        detection_accuracy: Detection accuracy for each budget
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot SNR vs photon budget
    ax1.semilogx(photon_budgets, snr_values, 'go-', linewidth=2, markersize=6)
    ax1.set_xlabel("Photon Budget", fontsize=12)
    ax1.set_ylabel("Signal-to-Noise Ratio", fontsize=12)
    ax1.set_title("SNR vs Photon Budget", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot detection accuracy vs photon budget
    ax2.semilogx(photon_budgets, detection_accuracy, 'mo-', linewidth=2, markersize=6)
    ax2.set_xlabel("Photon Budget", fontsize=12)
    ax2.set_ylabel("Detection Accuracy", fontsize=12)
    ax2.set_title("Detection Accuracy vs Photon Budget", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
