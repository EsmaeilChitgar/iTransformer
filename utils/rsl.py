"""Training-only residual-subspace objective for multivariate forecasting."""

import numpy as np
import torch


def training_basis(train_dataset, variance_ratio, device):
    """Build PCA from unique, scaled training timestamps (not sliding windows)."""
    if not 0.0 < variance_ratio < 1.0:
        raise ValueError('rsl_variance_ratio must be strictly between 0 and 1')
    if not hasattr(train_dataset, 'data_x') or train_dataset.set_type != 0:
        raise ValueError('RSL requires a Dataset_Custom training split')

    x = np.asarray(train_dataset.data_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError('RSL needs at least two timestamps and two channels')
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    total = eigenvalues.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Train covariance has no finite positive variance')
    cumulative = np.cumsum(eigenvalues) / total
    rank = min(int(np.searchsorted(cumulative, variance_ratio) + 1), x.shape[1] - 1)
    basis = torch.from_numpy(eigenvectors[:, order[:rank]].copy()).to(
        device=device, dtype=torch.float32
    )
    print('RSL train timestamps: {} channels: {} rank: {} variance: {:.6f}'.format(
        x.shape[0], x.shape[1], rank, cumulative[rank - 1]
    ))
    return basis


def residual_loss(outputs, targets, basis):
    """Mean squared error per residual direction; no D x D projector."""
    error = (outputs - targets).float()
    d, r = error.shape[-1], basis.shape[-1]
    if basis.shape[0] != d or not 0 < r < d:
        raise ValueError('RSL basis dimension does not match forecast channels')
    # Compute in float32 even inside autocast; energy identity needs precision.
    with torch.cuda.amp.autocast(enabled=False):
        projected = error @ basis
        residual_energy = (error.square().sum() - projected.square().sum()).clamp_min(0)
        return residual_energy / (error.numel() // d * (d - r))
