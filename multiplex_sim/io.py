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

def safe_load_tab_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1].replace('%', ''))])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["wavelength", "intensity"])

def process_txt_pair(abs_path, em_path):
    try:
        abs_df = safe_load_tab_txt(abs_path)
        em_df = safe_load_tab_txt(em_path)

        wl_ex = abs_df["wavelength"].values
        ex = abs_df["intensity"].values
        wl_em = em_df["wavelength"].values
        em = em_df["intensity"].values

        return wl_ex, ex, wl_em, em
    except Exception as e:
        print(f"[TXT ERROR] {abs_path.name} + {em_path.name}: {e}")
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

        elif file_path.suffix.lower() == ".txt" and file_path.name.lower().endswith("abs.txt"):
            em_path = file_path.with_name(file_path.name.replace("Abs.txt", "Em.txt"))
            if em_path.exists():
                result = process_txt_pair(file_path, em_path)
                if result:
                    wl_ex, ex, wl_em, em = result
                    name = file_path.stem.replace("Abs", "")
                    save_npz(name, wl_ex, ex, wl_em, em, output_dir)
            else:
                print(f"[TXT PAIR MISSING] Could not find: {em_path.name}")

def list_fluorophores(npz_folder="spectra_npz"):
    """
    Lists all fluorophore .npz files in a given folder.
    
    Parameters:
        npz_folder (str): Directory containing .npz spectra files.
    
    Returns:
        List of fluorophore names (without extension).
    """
    folder = Path(npz_folder)
    if not folder.exists():
        print(f"[ERROR] Folder '{npz_folder}' does not exist.")
        return []

    files = sorted(folder.glob("*.npz"))
    return [f.stem for f in files]