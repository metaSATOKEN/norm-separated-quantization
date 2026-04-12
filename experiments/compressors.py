"""
Compression / Decompression primitives for Arc-Compression v2.0

All compressors follow the same interface:
    compressed = compress(hidden_states, basis, **kwargs)
    reconstructed = decompress(compressed, basis, **kwargs)

where hidden_states: (T, d_model) tensor
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CompressedState:
    """Compressed representation of a hidden state."""
    norms: torch.Tensor          # (T,) — per-token norms
    residual_proj: torch.Tensor  # (T, k) — projection onto basis
    method: str                  # compression method name
    k: int                       # rank of residual
    original_shape: tuple        # (T, d_model)


def compute_pca_basis(
    hidden_states: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute top-k PCA basis from norm-normalized hidden states.

    Args:
        hidden_states: (T, d_model)
        k: number of components

    Returns:
        basis: (k, d_model) orthonormal rows
    """
    h = hidden_states.float()
    norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
    h_normed = h / norms
    h_centered = h_normed - h_normed.mean(dim=0, keepdim=True)

    # SVD for PCA
    U, S, Vt = torch.linalg.svd(h_centered, full_matrices=False)
    basis = Vt[:k]  # (k, d_model)
    return basis


def compute_random_basis(
    d_model: int,
    k: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a random orthonormal basis (for baseline comparison).

    Returns:
        basis: (k, d_model) orthonormal rows
    """
    rng = torch.Generator().manual_seed(seed)
    M = torch.randn(k, d_model, generator=rng)
    Q, _ = torch.linalg.qr(M.T)
    return Q.T[:k]  # (k, d_model)


def compress_norm_only(
    hidden_states: torch.Tensor,
) -> CompressedState:
    """Extreme compression: keep only norms, discard all direction info."""
    norms = hidden_states.float().norm(dim=1)
    return CompressedState(
        norms=norms,
        residual_proj=torch.empty(0),
        method="norm_only",
        k=0,
        original_shape=hidden_states.shape,
    )


def decompress_norm_only(
    compressed: CompressedState,
    mean_direction: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct from norms only, using mean direction."""
    direction = mean_direction / mean_direction.norm().clamp(min=1e-12)
    return compressed.norms.unsqueeze(1) * direction.unsqueeze(0)


def compress_norm_pca(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
) -> CompressedState:
    """
    Compress: norm + PCA top-k projection of residual.

    Decomposition: h = norm * (mean_dir + basis^T @ coeffs)
    Actually: h = norm * direction, direction projected onto basis
    """
    h = hidden_states.float()
    norms = h.norm(dim=1)  # (T,)
    h_normed = h / norms.unsqueeze(1).clamp(min=1e-12)  # (T, d_model)

    # Project normalized vectors onto basis
    proj = h_normed @ basis.T  # (T, k)

    return CompressedState(
        norms=norms,
        residual_proj=proj,
        method="norm_pca",
        k=basis.shape[0],
        original_shape=hidden_states.shape,
    )


def decompress_norm_pca(
    compressed: CompressedState,
    basis: torch.Tensor,
    mean_direction: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct from norm + PCA residual.

    direction_approx = mean_dir + basis^T @ coeffs
    h_approx = norm * normalize(direction_approx)
    """
    # Reconstruct direction from projection
    direction_approx = compressed.residual_proj @ basis  # (T, d_model)
    # Add mean direction
    direction_approx = direction_approx + mean_direction.unsqueeze(0)
    # Re-normalize to unit sphere
    direction_approx = direction_approx / direction_approx.norm(
        dim=1, keepdim=True
    ).clamp(min=1e-12)
    # Scale by original norms
    return compressed.norms.unsqueeze(1) * direction_approx
