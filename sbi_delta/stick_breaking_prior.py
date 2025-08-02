"""Stick breaking prior for filter and gap allocation with variable start."""

import torch
from torch.distributions import Distribution, Beta, Bernoulli, Transform
import numpy as np
import matplotlib.pyplot as plt

class StickBreakingTransform(Transform):
    """Transform from unconstrained space to stick-breaking proportions."""
    
    def __init__(self, concentration=1.0):
        super().__init__()
        self.concentration = concentration
        
    def _call(self, x):
        # x: batch_shape x n_segments
        remaining = torch.ones_like(x[..., 0])
        proportions = []
        
        for prop in x[..., :-1].transpose(0, -1):  # Process all but last segment
            this_prop = remaining * prop
            proportions.append(this_prop)
            remaining = remaining - this_prop
        
        proportions.append(remaining)  # Last segment gets what's left
        return torch.stack(proportions, dim=-1)
        
    def _inverse(self, y):
        # y: batch_shape x n_segments
        remaining = torch.ones_like(y[..., 0])
        x = []
        
        for prop in y[..., :-1].transpose(0, -1):
            x.append(prop / remaining)
            remaining = remaining - prop
            
        return torch.stack(x + [y[..., -1]], dim=-1)

class StickBreakingPrior(Distribution):
    """
    Prior distribution for filter and gap widths using stick breaking process.
    Can start with either filter or gap, but always ends with a filter.
    
    Args:
        n_filters: Number of filters
        total_width: Total wavelength range to allocate
        min_filter_width: Minimum allowed filter width
        concentration: Beta distribution concentration parameter
        start_filter_prob: Probability of starting with a filter (default 0.5)
    """
    
    def __init__(self, n_filters, total_width, min_filter_width, max_filter_width, concentration=1.0, start_filter_prob=0.5):
        super().__init__()
        self.n_filters = n_filters
        self.n_segments = 2 * n_filters - 1  # n filters + (n-1) gaps
        self.total_width = total_width
        self.min_filter_width = min_filter_width
        self.max_filter_width = max_filter_width
        self.concentration = concentration
        
        # Start type prior (filter or gap)
        self.start_prior = Bernoulli(probs=start_filter_prob)
        
        # Initialize transform
        self.transform = StickBreakingTransform(concentration)
        
    def sample(self, sample_shape=torch.Size()):
        # Sample whether to start with filter (1) or gap (0)
        start_with_filter = self.start_prior.sample(sample_shape)
        
        # Base distribution is Beta for proportions
        base_dist = Beta(
            self.concentration * torch.ones(self.n_segments),
            self.concentration * torch.ones(self.n_segments)
        )
        
        # Sample proportions using stick breaking
        props = self.transform(base_dist.sample(sample_shape))
        
        # Convert proportions to actual widths
        # Convert proportions to widths, scaling between min and max width
        widths = props * (self.max_filter_width - self.min_filter_width) + self.min_filter_width
        
        # Create filter mask based on start type
        filter_mask = torch.zeros_like(widths)
        filter_indices = torch.arange(self.n_filters) * 2  # [0, 2, 4, ...]
        # Handle batched and non-batched cases
        start_filter_batch = start_with_filter.bool() if len(start_with_filter.shape) == 0 else start_with_filter
        if not start_filter_batch.any():
            filter_indices = filter_indices + 1  # [1, 3, 5, ...] if starting with gap
            
        # Set filter positions based on start type
        for batch_idx in torch.broadcast_tensors(torch.arange(sample_shape[0]) if sample_shape else torch.tensor([0]))[0]:
            filter_mask[batch_idx, filter_indices[filter_indices < self.n_segments]] = 1
            
        # Ensure minimum filter width constraint
        filters = widths * filter_mask
        gaps = widths * (1 - filter_mask)
        
        # Where filters are too narrow, take space from adjacent gaps
        for batch_idx in range(widths.shape[0] if sample_shape else 1):
            curr_filter_indices = torch.where(filter_mask[batch_idx])[0]
            for i in curr_filter_indices:
                if filters[batch_idx, i] < self.min_filter_width:
                    needed = self.min_filter_width - filters[batch_idx, i]
                    
                    # Take from gaps proportionally if available
                    available_gaps = []
                    if i > 0:  # Left gap
                        available_gaps.append((i-1, gaps[batch_idx, i-1]))
                    if i < self.n_segments-1:  # Right gap
                        available_gaps.append((i+1, gaps[batch_idx, i+1]))
                        
                    if available_gaps:
                        # Distribute needed width among available gaps
                        total_gap = sum(g[1] for g in available_gaps)
                        for gap_idx, gap_width in available_gaps:
                            take = needed * (gap_width / total_gap)
                            gaps[batch_idx, gap_idx] -= take
                            filters[batch_idx, i] += take
        
        # Combine back into single tensor
        result = torch.zeros_like(widths)
        result = result.masked_scatter_(filter_mask.bool(), filters[filter_mask.bool()])
        result = result.masked_scatter_((~filter_mask.bool()), gaps[~filter_mask.bool()])
        
        # Return both the start type and widths
        return torch.cat([start_with_filter.float().unsqueeze(-1), result], dim=-1)
    
    def log_prob(self, value):
        # Split into start type and widths
        start_with_filter = value[..., 0]
        widths = value[..., 1:]
        
        # Check start type probability
        start_log_prob = self.start_prior.log_prob(start_with_filter)
        
        # Convert widths back to proportions
        props = widths / self.total_width
        
        # Check constraints
        if not (props.sum(-1) <= 1.0).all():
            return -float('inf')
            
        # Create filter mask based on start type
        filter_mask = torch.zeros_like(widths)
        filter_indices = torch.arange(self.n_filters) * 2
        filter_indices = filter_indices + (1 - start_with_filter).long().unsqueeze(-1)
        filter_indices = filter_indices[filter_indices < self.n_segments]
        filter_mask.scatter_(-1, filter_indices.unsqueeze(0), 1)
            
        filter_widths = widths * filter_mask
        if not (filter_widths[filter_mask.bool()] >= self.min_filter_width).all():
            return -float('inf')
            
        # Get log prob from base distribution
        try:
            x = self.transform._inverse(props)
            base_dist = Beta(
                self.concentration * torch.ones_like(props),
                self.concentration * torch.ones_like(props)
            )
            return start_log_prob + base_dist.log_prob(x).sum(-1)
        except:
            return -float('inf')
            
    def visualize_samples(self, n_samples=5):
        """Visualize multiple samples from the prior to show filter/gap configurations.
        
        Args:
            n_samples: Number of configurations to plot
        """
        samples = self.sample((n_samples,))
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
            
        wavelengths = np.linspace(0, self.total_width, 1000)
        
        for i, (ax, sample) in enumerate(zip(axes, samples)):
            start_with_filter = bool(sample[0].item())
            widths = sample[1:].numpy()
            
            # Calculate segment positions
            positions = np.cumsum(np.concatenate([[0], widths]))
            
            # Plot each segment
            for j, (start, width) in enumerate(zip(positions[:-1], widths)):
                is_filter = (j % 2 == 0) if start_with_filter else (j % 2 == 1)
                color = 'blue' if is_filter else 'gray'
                alpha = 0.3 if is_filter else 0.15
                label = f'Filter {j//2 + 1}' if is_filter else f'Gap {j//2 + 1}'
                
                ax.fill_between([start, start + width], [0, 0], [1, 1], 
                               color=color, alpha=alpha, label=label)
                ax.text(start + width/2, 0.5, f'{width:.1f}nm', 
                       ha='center', va='center')
            
            # Customize plot
            ax.set_xlim(0, self.total_width)
            ax.set_ylim(0, 1)
            ax.set_title(f'Configuration {i+1}: ' + 
                        ('Starts with Filter' if start_with_filter else 'Starts with Gap'))
            ax.set_xlabel('Wavelength (nm)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        return fig, axes