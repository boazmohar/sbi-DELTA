import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm

def plot_fluorophores(names, npz_folder="spectra_npz", normalize=True):
    """
    Plot excitation (dashed) and emission (solid) spectra for multiple dyes.

    Parameters:
        names (list[str]): List of dye names (without .npz extension)
        npz_folder (str): Folder where .npz spectra files are stored
        normalize (bool): If True, peak-normalize each spectrum to max = 1
    """
    folder = Path(npz_folder)
    if not folder.exists():
        print(f"[ERROR] Folder '{npz_folder}' does not exist.")
        return

    n = len(names)
    colormap = cm.get_cmap("turbo", n)

    plt.figure(figsize=(10, 5))
    for idx, dye_name in enumerate(names):
        path = folder / f"{dye_name}.npz"
        if not path.exists():
            print(f"[WARN] Missing file: {path.name}")
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
        plt.plot(wl_em, em, label=f"{dye_name} Emission", color=color)
        plt.plot(wl_ex, ex, '--', label=f"{dye_name} Excitation", color=color)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity" if normalize else "Raw Intensity")
    plt.title("Excitation and Emission Spectra")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()