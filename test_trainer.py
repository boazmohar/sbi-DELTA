"""
Test for sbi_delta.trainer script and workflow.
Covers training, validation, and R^2 output.
"""
import os
import torch
import numpy as np
from sbi_delta.trainer import main as trainer_main

def test_trainer_runs_and_saves(tmp_path):
    # Patch save_dir to tmp_path
    import sbi_delta.trainer as trainer_mod
    orig_save_dir = "sbi_training_demo_results"
    trainer_mod.save_dir = str(tmp_path)
    trainer_main()
    # Check results file exists
    results_path = os.path.join(str(tmp_path), "results.pt")
    assert os.path.exists(results_path)
    results = torch.load(results_path)
    # Check keys and shapes
    assert "train_theta" in results and "train_x" in results
    assert "val_theta" in results and "val_x" in results
    assert "pred_theta" in results and "r2_scores" in results
    assert results["train_theta"].shape[0] > 100
    assert results["val_theta"].shape[0] > 10
    assert np.mean(results["r2_scores"]) > 0.0  # Should be non-trivial
