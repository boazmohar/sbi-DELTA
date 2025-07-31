"""
Test for sbi_delta.trainer script and workflow.
Covers training, validation, and R^2 output.
"""
import numpy as np

import pytest
from pathlib import Path

def create_npz(path, wl_em, em, wl_ex, ex):
    np.savez(path,
             wavelengths_emission=wl_em,
             emission=em,
             wavelengths_excitation=wl_ex,
             excitation=ex)

from sbi_delta.trainer import Trainer
from sbi_delta.simulator.emission_simulator import EmissionSimulator
from sbi_delta.spectra_manager import SpectraManager
from sbi_delta.filter_bank import FilterBank
from sbi_delta.config import BaseConfig
from sbi_delta.excitation_manager import ExcitationManager
from sbi_delta.config import ExcitationConfig
import types


def test_trainer_runs_and_saves(tmp_path):
    # Minimal config for test
    # Create synthetic spectra files
    data = tmp_path / "data"
    data.mkdir()
    wl = np.linspace(400, 700, 4)
    em1 = np.array([0.0, 2.0, 1.0, 0.0])
    ex1 = np.array([1.0, 0.5, 0.0, 0.0])
    em2 = np.array([0.0, 0.0, 2.0, 1.0])
    ex2 = np.array([0.0, 1.0, 0.5, 0.0])
    create_npz(data / "dye1.npz", wl, em1, wl, ex1)
    create_npz(data / "dye2.npz", wl, em2, wl, ex2)
    config = BaseConfig(
        min_wavelength=400.0,
        max_wavelength=700.0,
        wavelength_step=100.0,
        spectra_folder=data,
        dye_names=["dye1", "dye2"],
        bg_dye=None
    )
    spectra_manager = SpectraManager(config)
    spectra_manager.load()
    from sbi_delta.config import FilterConfig
    filter_cfgs = [
        FilterConfig(start=400.0, stop=550.0, sharpness=10.0),
        FilterConfig(start=550.0, stop=700.0, sharpness=10.0)
    ]
    filter_bank = FilterBank(config, configs=filter_cfgs)
    excitation_cfg = ExcitationConfig(excitation_mode="manual", manual_wavelengths=[500.0, 600.0])
    excitation_manager = ExcitationManager(config, excitation_cfg=excitation_cfg, spectra_manager=spectra_manager)
    simulator = EmissionSimulator(spectra_manager, filter_bank, config)
    simulator.excitation_manager = excitation_manager
    trainer = Trainer(simulator, n_train=20, n_val=5, save_dir=str(tmp_path))
    posterior = trainer.train()
    assert posterior is not None
    r2_scores, rmse_scores, rmse = trainer.validate(n_samples=10)
    # Check keys and shapes
    results = trainer.results
    assert "train_theta" in results and "train_x" in results
    assert "val_theta" in results and "val_x" in results
    assert "pred_theta" in results and "r2_scores" in results
    assert results["train_theta"].shape[0] > 1
    assert results["val_theta"].shape[0] > 1
    assert np.all(np.isfinite(results["r2_scores"])), "R² contains NaN or inf"


def test_trainer_with_custom_network(tmp_path):
    """
    Test Trainer with a custom network architecture argument (e.g., different density estimator).
    """
    # Create synthetic spectra files
    data = tmp_path / "data"
    data.mkdir()
    wl = np.linspace(400, 700, 4)
    em1 = np.array([0.0, 2.0, 1.0, 0.0])
    ex1 = np.array([1.0, 0.5, 0.0, 0.0])
    em2 = np.array([0.0, 0.0, 2.0, 1.0])
    ex2 = np.array([0.0, 1.0, 0.5, 0.0])
    create_npz(data / "dye1.npz", wl, em1, wl, ex1)
    create_npz(data / "dye2.npz", wl, em2, wl, ex2)
    config = BaseConfig(
        min_wavelength=400.0,
        max_wavelength=700.0,
        wavelength_step=100.0,
        spectra_folder=data,
        dye_names=["dye1", "dye2"],
        bg_dye=None
    )
    spectra_manager = SpectraManager(config)
    spectra_manager.load()
    from sbi_delta.config import FilterConfig
    filter_cfgs = [
        FilterConfig(start=400.0, stop=550.0, sharpness=10.0),
        FilterConfig(start=550.0, stop=700.0, sharpness=10.0)
    ]
    filter_bank = FilterBank(config, configs=filter_cfgs)
    excitation_cfg = ExcitationConfig(excitation_mode="manual", manual_wavelengths=[500.0, 600.0])
    excitation_manager = ExcitationManager(config, excitation_cfg=excitation_cfg, spectra_manager=spectra_manager)
    simulator = EmissionSimulator(spectra_manager, filter_bank, config)
    simulator.excitation_manager = excitation_manager
    network_architecture = {'density_estimator': 'maf'}
    trainer = Trainer(simulator, n_train=20, n_val=5, save_dir=str(tmp_path), network_architecture=network_architecture)
    posterior = trainer.train()
    assert posterior is not None
    r2_scores, rmse_scores, rmse = trainer.validate(n_samples=10)
    assert np.all(np.isfinite(r2_scores)), "R² contains NaN or inf"


def test_trainer_network_architecture_effect(tmp_path):
    """
    Test that different network architectures produce different results on the same synthetic data.
    """
    # Create synthetic spectra files
    data = tmp_path / "data"
    data.mkdir()
    wl = np.linspace(400, 700, 4)
    em1 = np.array([0.0, 2.0, 1.0, 0.0])
    ex1 = np.array([1.0, 0.5, 0.0, 0.0])
    em2 = np.array([0.0, 0.0, 2.0, 1.0])
    ex2 = np.array([0.0, 1.0, 0.5, 0.0])
    create_npz(data / "dye1.npz", wl, em1, wl, ex1)
    create_npz(data / "dye2.npz", wl, em2, wl, ex2)
    config = BaseConfig(
        min_wavelength=400.0,
        max_wavelength=700.0,
        wavelength_step=100.0,
        spectra_folder=data,
        dye_names=["dye1", "dye2"],
        bg_dye=None
    )
    spectra_manager = SpectraManager(config)
    spectra_manager.load()
    from sbi_delta.config import FilterConfig
    filter_cfgs = [
        FilterConfig(start=400.0, stop=550.0, sharpness=10.0),
        FilterConfig(start=550.0, stop=700.0, sharpness=10.0)
    ]
    filter_bank = FilterBank(config, configs=filter_cfgs)
    excitation_cfg = ExcitationConfig(excitation_mode="manual", manual_wavelengths=[500.0, 600.0])
    excitation_manager = ExcitationManager(config, excitation_cfg=excitation_cfg, spectra_manager=spectra_manager)
    simulator = EmissionSimulator(spectra_manager, filter_bank, config)
    simulator.excitation_manager = excitation_manager

    # Run with two different architectures
    arch1 = {'density_estimator': 'maf'}
    arch2 = {'density_estimator': 'nsf'}
    trainer1 = Trainer(simulator, n_train=20, n_val=5, save_dir=str(tmp_path / "maf"), network_architecture=arch1)
    posterior1 = trainer1.train()
    r2_scores1, _, _ = trainer1.validate(n_samples=10)
    trainer2 = Trainer(simulator, n_train=20, n_val=5, save_dir=str(tmp_path / "nsf"), network_architecture=arch2)
    posterior2 = trainer2.train()
    r2_scores2, _, _ = trainer2.validate(n_samples=10)
    # Check that the results are not identical (network architecture has an effect)
    assert not np.allclose(r2_scores1, r2_scores2), "Different architectures should yield different R² scores"

    # Check that the posteriors are not identical by comparing samples
    # Draw samples from both posteriors at the same observation
    # Use the first validation observation from trainer1
    obs = trainer1.results["val_x"][0]
    samples1 = posterior1.sample((100,), x=obs).numpy()
    samples2 = posterior2.sample((100,), x=obs).numpy()
    mean_samples1 = np.mean(samples1, axis=0)
    mean_samples2 = np.mean(samples2, axis=0)
    # Check that the means are not all close
    assert not np.allclose(mean_samples1, mean_samples2), "Different architectures should yield different posterior means"
    # Check that the samples are not all close
    assert not np.allclose(samples1, samples2), "Different architectures should yield different posteriors (samples)"
 