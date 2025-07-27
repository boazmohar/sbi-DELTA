import pytest
import torch
from sbi_delta.config import PriorConfig, BaseConfig
from sbi_delta.prior_manager import PriorManager
from torch.distributions import Dirichlet, Uniform

@pytest.fixture
def base_config():
    return BaseConfig(dye_names=["A", "B", "C"], bg_dye="AF_v1")

@pytest.fixture
def prior_manager(base_config):
    prior_config = PriorConfig(dirichlet_concentration=2.0, include_background_ratio=True, background_ratio_bounds=(0.1, 0.9))
    return PriorManager(prior_config, base_config)

def test_get_concentration_prior(prior_manager):
    prior = prior_manager.get_concentration_prior()
    assert isinstance(prior, torch.distributions.Dirichlet)
    assert prior.concentration.shape[0] == 3
    samples = prior.sample((100,))
    assert samples.shape == (100, 3)
    assert torch.allclose(samples.sum(dim=1), torch.ones(100), atol=1e-5)

def test_get_background_ratio_prior(prior_manager):
    prior = prior_manager.get_background_ratio_prior()
    assert isinstance(prior, torch.distributions.Uniform)
    samples = prior.sample((100,))
    assert samples.min() >= 0.1 and samples.max() <= 0.9

def test_get_joint_prior(prior_manager):
    joint = prior_manager.get_joint_prior()
    samples = joint.sample((50,))
    assert samples.shape == (50, 4)  # 3 concentrations + 1 background
    # Check sum of concentrations is 1
    assert torch.allclose(samples[:, :3].sum(dim=1), torch.ones(50), atol=1e-5)
    # Check background in bounds
    assert (samples[:, 3] >= 0.1).all() and (samples[:, 3] <= 0.9).all()
    # Check log_prob shape and values
    logp = joint.log_prob(samples)
    assert logp.shape == (50,)
    # log_prob should be finite and match sum of individual priors
    d = Dirichlet(torch.full((3,), 2.0))
    u = Uniform(0.1, 0.9)
    expected_logp = d.log_prob(samples[:, :3]) + u.log_prob(samples[:, 3])
    assert torch.allclose(logp, expected_logp, atol=1e-5)
    # Test arg_constraints
    ac = joint.arg_constraints
    assert isinstance(ac, dict)
   

def test_visualize_dirichlet_prior(prior_manager):
    # Should not raise
    ax = prior_manager.visualize_dirichlet_prior(n_samples=100)
    assert ax is not None

def test_prior_manager_requires_bg_dye():
    from sbi_delta.config import PriorConfig, BaseConfig
    base_config = BaseConfig(dye_names=["A", "B", "C"], bg_dye=None)
    prior_config = PriorConfig(dirichlet_concentration=2.0, include_background_ratio=True, background_ratio_bounds=(0.1, 0.9))
    with pytest.raises(ValueError, match="must set bg_dye"):
        PriorManager(prior_config, base_config)