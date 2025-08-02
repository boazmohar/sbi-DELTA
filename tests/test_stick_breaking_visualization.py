"""Test visualization of stick breaking prior."""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbi_delta.stick_breaking_prior import StickBreakingPrior

def test_filter_configurations():
    """Generate and visualize different filter configurations."""
    
    # Create prior with 3 filters over 300nm range
    prior = StickBreakingPrior(
        n_filters=3,
        total_width=300,  # e.g., 450-750nm
        min_filter_width=10,
        concentration=1.0,
        start_filter_prob=0.5
    )
    
    # Generate and plot samples
    fig, axes = prior.visualize_samples(n_samples=6)  # Show both filter-first and gap-first examples
    fig.suptitle('Example Filter Configurations\nBlue = Filters, Gray = Gaps')
    fig.savefig('filter_configurations.png', bbox_inches='tight', dpi=150)

if __name__ == "__main__":
    test_filter_configurations()