import pandas as pd
import numpy as np
from pathlib import Path
import os

def process_csv(file_path: Path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip().lower() for col in df.columns]
        wl = df['wavelength'].values
        em = df.get('emission', np.zeros_like(wl))
        ex = df.get('excitation', np.zeros_like(wl))
        return wl, ex, em
    except Exception as e:
        print(f"[CSV ERROR] {file_path.name}: {e}")
        return None

def process_xlsx(file_path: Path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        df.columns = [col.strip() for col in df.columns]
        df = df.dropna(how='all')

        ex_wl = df.filter(like='Wavelength', axis=1).iloc[:, 0].dropna()
        ex = df.filter(like='ex', axis=1).iloc[:, 0].dropna()
        em_wl = df.filter(like='Wavelength', axis=1).iloc[:, -1].dropna()
        em = df.filter(like='em', axis=1).iloc[:, 0].dropna()

        return ex_wl.values, ex.values, em_wl.values, em.values
    except Exception as e:
        print(f"[XLSX ERROR] {file_path.name}: {e}")
        return None

def save_npz(name: str, wl_ex, ex, wl_em, em, out_dir: Path):
    npz_path = out_dir / f"{name}.npz"
    np.savez(npz_path,
             wavelengths_excitation=wl_ex,
             excitation=ex,
             wavelengths_emission=wl_em,
             emission=em)
    print(f"[SAVED] {npz_path.name}")

def process_spectra_folder(folder_path: str, out_folder: str = "npz_output"):
    input_dir = Path(folder_path)
    output_dir = Path(out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.rglob("*"):
        if file_path.suffix.lower() == ".csv":
            result = process_csv(file_path)
            if result:
                wl, ex, em = result
                save_npz(file_path.stem, wl, ex, wl, em, output_dir)

        elif file_path.suffix.lower() in [".xls", ".xlsx"]:
            result = process_xlsx(file_path)
            if result:
                wl_ex, ex, wl_em, em = result
                save_npz(file_path.stem, wl_ex, ex, wl_em, em, output_dir)
