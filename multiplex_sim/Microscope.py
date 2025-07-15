import numpy as np
import torch

def gaussian_emission(wavelengths: np.ndarray, λ_max: float, σ: float) -> np.ndarray:
    """Generate Gaussian emission spectrum."""
    return np.exp(-0.5 * ((wavelengths - λ_max) / σ)**2)

import numpy as np
import torch

def gaussian_emission(wavelengths: np.ndarray, λ_max: float, σ: float) -> np.ndarray:
    """Generate Gaussian emission spectrum."""
    return np.exp(-0.5 * ((wavelengths - λ_max) / σ)**2)

def simulate_detected_signal(
    params: torch.Tensor,
    num_channels: int = 2,
    center_wavelengths: list = None,
    bandwidth: float = 30.0
) -> torch.Tensor:
    """
    Simulate signal across multiple channels with narrow-band Gaussian filters.

    params: Tensor of shape (batch_size, 2), with [λ_max, σ]
    num_channels: number of detection channels
    center_wavelengths: list of peak wavelengths for each channel (if None, evenly spaced)
    bandwidth: standard deviation of each channel's filter response (in nm)

    Returns: Tensor of shape (batch_size, num_channels)
    """
    λ = np.arange(400, 700, 1)

    if center_wavelengths is None:
        center_wavelengths = np.linspace(420, 680, num_channels)

    channel_filters = [np.exp(-0.5 * ((λ - cw) / bandwidth)**2) for cw in center_wavelengths]
    channel_filters = np.stack(channel_filters)  # shape: (num_channels, len(λ))

    signals = []
    for p in params:
        λ_max, σ = p.numpy()
        emission = gaussian_emission(λ, λ_max, σ)
        signal_vector = np.sum(channel_filters * emission, axis=1)  # shape: (num_channels,)
        signals.append(signal_vector)

    return torch.tensor(signals, dtype=torch.float32)

import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

def load_excitation_spectra(fluor_names, spectra_dir):
    spectra_dir = Path(spectra_dir)
    spectra = {}
    for name in fluor_names:
        path = spectra_dir / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            wl = data["wavelengths_excitation"]
            excitation = data["excitation"]
            excitation /= excitation.max()
            spectra[name] = (wl, excitation)
        else:
            print(f"[WARNING] Missing {name}.npz")
    return spectra

def excitation_cost(wavelengths, spectra_dict, shared_wl, verbose=False):
    total_cost = 0
    for i, (name_i, λi) in enumerate(zip(spectra_dict.keys(), wavelengths)):
        _, ex_i = spectra_dict[name_i]
        idx_i = np.clip(np.searchsorted(shared_wl, λi, side="left"), 0, len(ex_i) - 1)
        self_signal = ex_i[idx_i]

        crosstalk = 0
        for j, name_j in enumerate(spectra_dict.keys()):
            if i == j:
                continue
            _, ex_j = spectra_dict[name_j]
            idx_j = np.clip(np.searchsorted(shared_wl, λi, side="left"), 0, len(ex_j) - 1)
            crosstalk += ex_j[idx_j]

        if verbose:
            print(f"{name_i} @ {λi:.1f} nm → self: {self_signal:.3f}, crosstalk: {crosstalk:.3f}, net: {-self_signal + crosstalk:.3f}")

        total_cost += -self_signal + crosstalk

    return total_cost

def find_optimal_excitation(fluor_names, spectra_dir):
    spectra = load_excitation_spectra(fluor_names, spectra_dir)
    if len(spectra) != len(fluor_names):
        missing = set(fluor_names) - set(spectra.keys())
        raise ValueError(f"Missing spectra for: {missing}")

    shared_wavelength = list(spectra.values())[0][0]
    shared_wavelength = np.asarray(shared_wavelength)
    bounds = []
    for name in fluor_names:
        wl, ex = spectra[name]
        max_idx = np.argmax(ex)
        peak_wl = wl[max_idx]
        lower = max(wl[0], peak_wl - 30)
        upper = min(wl[-1], peak_wl + 30)
        bounds.append((lower, upper))

    # bounds = [(shared_wavelength[0], shared_wavelength[-1])] * len(fluor_names)
    result = differential_evolution(
        func=lambda x, *a: excitation_cost(x, *a, verbose=True),
        bounds=bounds,
        args=(spectra, shared_wavelength),
        strategy='best1bin',
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        seed=42
    ).x

    return dict(zip(fluor_names, np.round(result).astype(int)))