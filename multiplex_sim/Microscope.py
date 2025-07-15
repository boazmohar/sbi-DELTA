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